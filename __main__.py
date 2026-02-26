"""Package entry point: python -m thirdparty.generate_caption {caption|summarize}"""
import sys

COMMANDS = ('caption', 'summarize')

if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    print(f"Usage: python -m thirdparty.generate_caption {{{','.join(COMMANDS)}}} [args...]")
    sys.exit(1)

command = sys.argv.pop(1)
if command == 'caption':
    from thirdparty.generate_caption.generate_caption import main
elif command == 'summarize':
    from thirdparty.generate_caption.summarize import main

main()
