import warnings

DEBUG = True


def set_debug(debug, show_warnings):
    global DEBUG
    DEBUG = debug

    warnings.filterwarnings('default' if show_warnings else 'ignore')


def debug(*values):
    global DEBUG
    if(DEBUG):
        print(*values)
