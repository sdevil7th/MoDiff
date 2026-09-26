"""Immutable installer-tool identities shared by setup and runtime overlays."""

from types import MappingProxyType


UV_TOOL_LOCKS = MappingProxyType(
    {
        ("linux", "x86_64"): MappingProxyType(
            {
                "url": "https://github.com/astral-sh/uv/releases/download/0.11.26/uv-x86_64-unknown-linux-gnu.tar.gz",
                "archiveSha256": "6426a73c3837e6e2483ee344cbc00f36394d179afcba6183cb77437e67db4af0",
                "executable": "uv-x86_64-unknown-linux-gnu/uv",
                "executableSha256": "29b90e884c384e1578ac37335521d807c192aa44d5a4a9b9f4690bb3850e179d",
            }
        ),
        ("linux", "arm64"): MappingProxyType(
            {
                "url": "https://github.com/astral-sh/uv/releases/download/0.11.26/uv-aarch64-unknown-linux-gnu.tar.gz",
                "archiveSha256": "befa1a59c91e96eb601b0fd9a97c03dd666f17baba644b2b4db9c59a767e387e",
                "executable": "uv-aarch64-unknown-linux-gnu/uv",
                "executableSha256": "9a36adc1a125e969a6952ef69b8072960a532f45e3434b972250e61801861c5b",
            }
        ),
        ("macos", "x86_64"): MappingProxyType(
            {
                "url": "https://github.com/astral-sh/uv/releases/download/0.11.26/uv-x86_64-apple-darwin.tar.gz",
                "archiveSha256": "922b460202707dd5f4ccacbadbe7f6a546cc46e82a99bf50ca99a7977a78eddd",
                "executable": "uv-x86_64-apple-darwin/uv",
                "executableSha256": "ef0df4073dd04f3827b40c55ecb9c99144598a4eec728dd109d76fd7bead0375",
            }
        ),
        ("macos", "arm64"): MappingProxyType(
            {
                "url": "https://github.com/astral-sh/uv/releases/download/0.11.26/uv-aarch64-apple-darwin.tar.gz",
                "archiveSha256": "8f7fbf1708399b921857bce71e1d60f0d3ccf52a30caebc1c1a2f175dce13ab6",
                "executable": "uv-aarch64-apple-darwin/uv",
                "executableSha256": "c9300ed8425e2c85230259a172066a32b475bc56f7ebe907783b2459159ea554",
            }
        ),
        ("windows", "x86_64"): MappingProxyType(
            {
                "url": "https://github.com/astral-sh/uv/releases/download/0.11.26/uv-x86_64-pc-windows-msvc.zip",
                "archiveSha256": "4e1278ede866be6c0bf32d2f466cc6de7a9fb399ecf20c9ce2d186e52424be47",
                "executable": "uv.exe",
                "executableSha256": "deeaa21aac3e3e40b3fa00788208aa9a319cefbb3c2aa598cf580565a82ebc34",
            }
        ),
        ("windows", "arm64"): MappingProxyType(
            {
                "url": "https://github.com/astral-sh/uv/releases/download/0.11.26/uv-aarch64-pc-windows-msvc.zip",
                "archiveSha256": "98246149741f558e25e45ecf2b0b20f34de0634269f2bf0dcb4012d4b6ba289a",
                "executable": "uv.exe",
                "executableSha256": "f13a990b845aba00a30734c6c678e71b321148fdf8e28101033cce4d2b7452c5",
            }
        ),
    }
)
