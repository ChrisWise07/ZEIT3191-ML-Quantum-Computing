def file_handler(path, mode, func):
    try:
        with open(path, mode) as f:
            return func(f)
    except FileNotFoundError:
        return 0
