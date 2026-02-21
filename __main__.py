"""Package entry point: python -m generate_caption {caption|summarize}"""
import sys

COMMANDS = ('caption', 'summarize')

if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    print(f"Usage: python -m generate_caption {{{','.join(COMMANDS)}}} [args...]")
    sys.exit(1)

command = sys.argv.pop(1)
if command == 'caption':
    from .generate_caption import main
elif command == 'summarize':
    from .summarize import main

main()
