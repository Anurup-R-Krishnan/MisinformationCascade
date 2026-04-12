"""Root server app shim for OpenEnv validator compatibility."""

from misinformation_cascade_env.server.app import app as app
from misinformation_cascade_env.server.app import main as _pkg_main


def main(host: str = "0.0.0.0", port: int = 8000):
    return _pkg_main(host=host, port=port)


if __name__ == "__main__":
    main()
