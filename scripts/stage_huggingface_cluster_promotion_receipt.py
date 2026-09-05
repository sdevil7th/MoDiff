#!/usr/bin/env python3
"""Stage an exact Cluster promotion-v2 receipt after explicit human approval.

This command never edits the checked-in authority ledger.  It validates one
current frontend provenance document and its exact generated media bytes, then
prints either a receipt candidate or a complete ledger candidate for review.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.huggingface_cluster_promotions import (
    ClusterPromotionReceiptError,
    build_cluster_promotion_ledger_candidate,
    build_cluster_promotion_receipt_candidate,
    canonical_content_hash,
)
from modiff.legacy_cluster_compiler_mappings import (
    canonical_content_hash as compiler_mapping_content_hash,
    reviewed_historical_compiler_mappings,
    validate_historical_compiler_mappings,
)


_MAX_PROVENANCE_BYTES = 4 * 1024 * 1024
_MAX_CANDIDATE_REPORT_BYTES = 16 * 1024 * 1024
_CONFIRMATIONS = {
    "insertedThroughFrontend": "confirm_inserted_through_frontend",
    "expandedOfficialBlocks": "confirm_expanded_official_blocks",
    "editedParameters": "confirm_edited_parameters",
    "savedAndRefreshed": "confirm_saved_and_refreshed",
    "valuesRestoredExactly": "confirm_values_restored_exactly",
    "collapsedExpandedGraphEquivalent": "confirm_collapsed_expanded_graph_equivalent",
    "generatedThroughFrontend": "confirm_generated_through_frontend",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-id", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument(
        "--post-promotion-candidate-report",
        type=Path,
        required=True,
        help="Exact report emitted by the client compiler in post-promotion candidate mode.",
    )
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--receipt-id", required=True)
    parser.add_argument(
        "--reviewed-at",
        required=True,
        help="Explicit ISO-8601 review time with a UTC offset; no timestamp is inferred.",
    )
    parser.add_argument("--review-comment", required=True)
    parser.add_argument(
        "--approve-as-workspace-owner",
        action="store_true",
        help="Affirms that the workspace owner reviewed and approved this exact output.",
    )
    parser.add_argument("--confirm-inserted-through-frontend", action="store_true")
    parser.add_argument("--confirm-expanded-official-blocks", action="store_true")
    parser.add_argument("--confirm-edited-parameters", action="store_true")
    parser.add_argument("--confirm-saved-and-refreshed", action="store_true")
    parser.add_argument("--confirm-values-restored-exactly", action="store_true")
    parser.add_argument("--confirm-collapsed-expanded-graph-equivalent", action="store_true")
    parser.add_argument("--confirm-generated-through-frontend", action="store_true")
    output_kind = parser.add_mutually_exclusive_group()
    output_kind.add_argument(
        "--ledger-candidate",
        action="store_true",
        help="Print the complete, rehashed ledger candidate instead of only the receipt.",
    )
    output_kind.add_argument(
        "--atomic-bundle",
        action="store_true",
        help="Print a non-installing receipt, route-pin, and dependent-mapping candidate bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Create this new file exclusively. Existing files are never overwritten.",
    )
    return parser.parse_args(argv)


def _read_provenance(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ClusterPromotionReceiptError("Frontend provenance must be one regular file.")
    raw = path.read_bytes()
    if len(raw) > _MAX_PROVENANCE_BYTES:
        raise ClusterPromotionReceiptError("Frontend provenance exceeds the bounded size limit.")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ClusterPromotionReceiptError("Frontend provenance is not valid JSON.") from error
    if not isinstance(value, dict):
        raise ClusterPromotionReceiptError("Frontend provenance must be one JSON object.")
    return value


def _hash_media(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ClusterPromotionReceiptError("Promotion media must be one regular file.")
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_size += len(chunk)
    if byte_size <= 0:
        raise ClusterPromotionReceiptError("Promotion media is empty.")
    return digest.hexdigest(), byte_size


def _post_promotion_route(path: Path, admission_id: str, evidence_block: dict) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ClusterPromotionReceiptError(
            "Post-promotion compiler report must be one regular file."
        )
    raw = path.read_bytes()
    if len(raw) > _MAX_CANDIDATE_REPORT_BYTES:
        raise ClusterPromotionReceiptError("Post-promotion compiler report is too large.")
    try:
        report = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ClusterPromotionReceiptError(
            "Post-promotion compiler report is not valid JSON."
        ) from error
    if (
        not isinstance(report, dict)
        or report.get("schemaVersion") != 1
        or report.get("format") != "modiff.registered-block-v2-route-candidates.v1"
        or report.get("postPromotionAdmissionId") != admission_id
        or report.get("candidateSuccesses") != []
        or report.get("blocked") != []
    ):
        raise ClusterPromotionReceiptError(
            "Post-promotion compiler report boundary or admission is invalid."
        )
    pins = report.get("existingPins")
    matches = (
        [item for item in pins if isinstance(item, dict) and item.get("admissionId") == admission_id]
        if isinstance(pins, list)
        else []
    )
    if len(matches) != 1:
        raise ClusterPromotionReceiptError(
            "Post-promotion compiler report does not contain one exact route."
        )
    pin = matches[0]
    if (
        pin.get("previousCompiledDefinitionContentHash") != evidence_block.get("contentHash")
        or pin.get("previousCompiledDefinitionCanonicalSha256")
        != evidence_block.get("canonicalSha256")
    ):
        raise ClusterPromotionReceiptError(
            "Post-promotion compiler report was not derived from the evidence route pin."
        )
    required = {
        "definitionId": pin.get("definitionId"),
        "manifestContentHash": pin.get("definitionContentHash"),
        "admissionId": admission_id,
        "blockDefinition": {
            "definitionId": admission_id,
            "contentHash": pin.get("compiledDefinitionContentHash"),
            "canonicalSha256": pin.get("compiledDefinitionCanonicalSha256"),
        },
        "executionGraphHash": pin.get("executionGraphHash"),
        "interfaceHash": pin.get("interfaceHash"),
    }
    if (
        any(
            not isinstance(required[field], str) or not required[field]
            for field in (
                "definitionId",
                "manifestContentHash",
                "admissionId",
                "executionGraphHash",
                "interfaceHash",
            )
        )
        or any(
            not isinstance(required["blockDefinition"].get(field), str)
            or not required["blockDefinition"][field]
            for field in ("definitionId", "contentHash", "canonicalSha256")
        )
    ):
        raise ClusterPromotionReceiptError("Post-promotion compiler route identity is incomplete.")
    return required


def _historical_mapping_candidate(post_route: dict) -> dict | None:
    ledger = reviewed_historical_compiler_mappings()
    matches = [
        mapping
        for mapping in ledger["mappings"]
        if mapping["destination"]["executionAdmissionId"] == post_route["admissionId"]
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ClusterPromotionReceiptError(
            "Post-promotion route has ambiguous historical compiler mappings."
        )
    mapping = matches[0]
    mapping["destination"].update(
        {
            "manifestDefinitionId": post_route["definitionId"],
            "manifestContentHash": post_route["manifestContentHash"],
            "blockDefinitionId": post_route["blockDefinition"]["definitionId"],
            "blockDefinitionContentHash": post_route["blockDefinition"]["contentHash"],
            "blockDefinitionCanonicalSha256": post_route["blockDefinition"][
                "canonicalSha256"
            ],
            "executionGraphHash": post_route["executionGraphHash"],
            "interfaceHash": post_route["interfaceHash"],
        }
    )
    mapping["mappingHash"] = compiler_mapping_content_hash(mapping, omit="mappingHash")
    ledger["contentHash"] = compiler_mapping_content_hash(ledger)
    return validate_historical_compiler_mappings(ledger)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    provenance = _read_provenance(args.provenance)
    media_sha256, media_byte_size = _hash_media(args.media)
    evidence_route = provenance.get("routeBinding")
    evidence_block = (
        evidence_route.get("blockDefinition")
        if isinstance(evidence_route, dict)
        else None
    )
    if not isinstance(evidence_block, dict):
        raise ClusterPromotionReceiptError("Frontend provenance route binding is missing.")
    post_route = _post_promotion_route(
        args.post_promotion_candidate_report,
        args.admission_id,
        evidence_block,
    )
    lifecycle = {field: getattr(args, argument) for field, argument in _CONFIRMATIONS.items()}
    receipt = build_cluster_promotion_receipt_candidate(
        admission_id=args.admission_id,
        provenance=provenance,
        media_sha256=media_sha256,
        media_byte_size=media_byte_size,
        receipt_id=args.receipt_id,
        reviewed_at=args.reviewed_at,
        review_comment=args.review_comment,
        lifecycle_confirmations=lifecycle,
        workspace_owner_approved=args.approve_as_workspace_owner,
        post_promotion_block_definition=post_route["blockDefinition"],
        post_promotion_manifest_content_hash=post_route["manifestContentHash"],
        output_index=args.output_index,
    )
    ledger_candidate = build_cluster_promotion_ledger_candidate(receipt)
    if args.atomic_bundle:
        document = {
            "schemaVersion": 1,
            "kind": "huggingface_cluster_post_promotion_atomic_candidate",
            "boundary": {
                "nonAuthorizingCandidate": True,
                "doesNotModifyWorkspace": True,
                "requiresAtomicCodeReview": True,
                "requiresFreshFrontendEvidence": True,
                "requiresExplicitWorkspaceOwnerApproval": True,
            },
            "admissionId": args.admission_id,
            "evidenceRouteBinding": deepcopy(evidence_route),
            "postPromotionRoute": post_route,
            "promotionReceipt": receipt,
            "promotionLedgerCandidate": ledger_candidate,
            "historicalCompilerMappingLedgerCandidate": _historical_mapping_candidate(
                post_route
            ),
        }
        document["contentHash"] = canonical_content_hash(document)
    elif args.ledger_candidate:
        document = ledger_candidate
    else:
        document = receipt
    encoded = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        if not args.output.parent.is_dir() or args.output.parent.is_symlink():
            raise ClusterPromotionReceiptError(
                "Promotion candidate output parent must be an existing regular directory."
            )
        with args.output.open("x", encoding="utf-8") as target:
            target.write(encoded)
        print(f"Staged non-authorizing promotion candidate at {args.output}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClusterPromotionReceiptError as error:
        raise SystemExit(str(error)) from error
