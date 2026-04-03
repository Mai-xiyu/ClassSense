# -*- coding: utf-8 -*-
"""启动入口"""

import uvicorn
from app.config import HOST, PORT


def main():
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
