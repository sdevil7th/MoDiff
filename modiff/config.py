# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import configparser
import logging
import os
from pathlib import Path
from socket import gethostbyname, gethostname

from modiff.secret_config import huggingface_token

# Define color codes
class ColorCodes:
    GREY = "\x1b[38;20m"
    BOLD_GREY = "\x1b[38;1m"
    DARK_GREY = "\x1b[30;20m"
    YELLOW = "\x1b[33;20m"
    BOLD_YELLOW = "\x1b[33;1m"
    RED = "\x1b[31;20m"
    BOLD_MAGENTA = "\x1b[35;1m"
    BLUE = "\x1b[34;20m"
    RESET = "\x1b[0m"
    GREEN = "\x1b[32;20m"

# Create custom formatter
class ColorFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: ColorCodes.DARK_GREY + "%(asctime)s [DEBUG] %(message)s" + ColorCodes.RESET,
        logging.INFO: ColorCodes.GREY + "%(asctime)s [INFO] %(message)s" + ColorCodes.RESET,
        logging.WARNING: ColorCodes.YELLOW + "%(asctime)s [WARN] %(message)s" + ColorCodes.RESET,
        logging.ERROR: ColorCodes.RED + "%(asctime)s [ERR] %(message)s" + ColorCodes.RESET,
        logging.CRITICAL: ColorCodes.BOLD_MAGENTA + "%(asctime)s [CRIT] %(message)s" + ColorCodes.RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y%m%d %H.%M.%S')
        return formatter.format(record)

class Config:
    def __init__(self, config_path: str | os.PathLike | None = None, dotenv_path: str | os.PathLike | None = None):
        app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = Path(config_path) if config_path is not None else Path(app_root) / 'config.ini'
        dotenv_path = Path(dotenv_path) if dotenv_path is not None else Path(app_root) / '.env'
        cfg = configparser.ConfigParser()
        cfg.optionxform = str  # disable lowercasing of keys
        cfg.read(config_path)

        self.server = {
            'host': cfg.get('server', 'host', fallback='127.0.0.1'),
            'port': cfg.getint('server', 'port', fallback=8088),
            'cors': cfg.getboolean('server', 'cors', fallback=False),
            'secure': cfg.getboolean('server', 'secure', fallback=False),
            'certfile': cfg.get('server', 'ssl_cert', fallback=None),
            'keyfile': cfg.get('server', 'ssl_key', fallback=None),
            # Keep the default aligned with config.example.ini.  The previous
            # 1 TiB fallback effectively disabled aiohttp's request-body guard
            # and allowed a single client to exhaust process memory or disk.
            'client_max_size': cfg.getint('server', 'client_max_size', fallback=1024**3),
        }
        if self.server['certfile'] and self.server['keyfile']:
            if not os.path.exists(self.server['certfile']):
                self.server['certfile'] = os.path.join(app_root, self.server['certfile'])
            if not os.path.exists(self.server['keyfile']):
                self.server['keyfile'] = os.path.join(app_root, self.server['keyfile'])

        cors_routes = cfg.get('server', 'cors_routes', fallback='*')
        self.server['cors_routes'] = [cors_route.strip() for cors_route in cors_routes.split(',')]
        self.server['ip'] = self.server['host'] if self.server['host'] != '0.0.0.0' else gethostbyname(gethostname())
        self.server['scheme'] = 'https' if self.server['secure'] else 'http'

        self.log = {
            'level': getattr(logging, cfg.get('logging', 'level', fallback='INFO').upper()),
        }

        hf_token, hf_token_source = huggingface_token(cfg, dotenv_path)
        self.hf = {
            'token': hf_token,
            'token_source': hf_token_source,
            'cache_dir': cfg.get('huggingface', 'cache_dir', fallback=None),
            'online_status': cfg.get('huggingface', 'online_status', fallback='Auto'),
        }
        if self.hf['cache_dir'] == '':
            self.hf['cache_dir'] = None
        if not self.hf['online_status'] in ['Auto', 'Online', 'Offline']:
            self.hf['online_status'] = 'Auto'

        self.paths = {
            'app_root': app_root,
            'work_dir': cfg.get('paths', 'work_dir', fallback=os.path.join(app_root, 'data')),
            'data': cfg.get('paths', 'data', fallback=os.path.join(app_root, 'data')),
            'images': cfg.get('paths', 'images', fallback=os.path.join(app_root, 'data', 'images')),
            'videos': cfg.get('paths', 'videos', fallback=os.path.join(app_root, 'data', 'videos')),
            'audio': cfg.get('paths', 'audio', fallback=os.path.join(app_root, 'data', 'audio')),
            'models': cfg.get('paths', 'models', fallback=os.path.join(app_root, 'data', 'models')),
            'upscalers': cfg.get('paths', 'upscalers', fallback=os.path.join(app_root, 'data', 'models', 'upscalers')),
            #'preprocessors': cfg.get('paths', 'preprocessors', fallback=os.path.join(app_root, 'data', 'preprocessors')),
            'temp': cfg.get('paths', 'temp', fallback=os.path.join(app_root, 'data', 'temp')),
        }
        if self.paths['work_dir'] == '~':
            self.paths['work_dir'] = str(Path.home())
        elif self.paths['work_dir'] == '':
            self.paths['work_dir'] = os.path.join(app_root, 'data')
        elif not os.path.isabs(self.paths['work_dir']):
            self.paths['work_dir'] = os.path.join(app_root, self.paths['work_dir'])

        if self.paths['data'] == '':
            self.paths['data'] = self.paths['work_dir']
        if self.paths['temp'] == '':
            self.paths['temp'] = os.path.join(self.paths['data'], 'temp')
        if self.paths['upscalers'] == '':
            self.paths['upscalers'] = os.path.join(self.paths['data'], 'upscalers')

        # Make sure all paths are absolute
        for path, value in self.paths.items():
            if not os.path.isabs(value):
                self.paths[path] = os.path.join(self.paths['work_dir'], value)

            if not os.path.exists(self.paths[path]) and path != 'work_dir':
                os.makedirs(self.paths[path])

        # Sets the environment variables
        self.environ = cfg['environ'] if 'environ' in cfg else {}
        for key, value in self.environ.items():
            self.environ[key] = value or ''
            os.environ[key] = value or ''

# Create logger
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter())
logger = logging.getLogger('modiff')
logger.propagate = False
logger.addHandler(handler)

# Load config
CONFIG = Config()

# Set logger level
logger.setLevel(CONFIG.log['level'])
