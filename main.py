# main.py
import asyncio
from modules.scanner import Scanner

def main():
    scanner = Scanner()
    asyncio.run(scanner.run())

if __name__ == "__main__":
    main()
