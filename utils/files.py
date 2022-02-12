import subprocess
from functools import reduce


def call_for_ret_code(args, silent=False):
    """
        Calls the subprocess and returns the return code
        :param args: list, arguments to fed into subprocess.call
        :param silent: bool, whether to display the ouput of the call
                       in stdout
        :returns int, 1 for failure and 0 for success, -1 for not found
    """
    if not silent:
        print("[+] " + reduce(lambda x, y: str(x) + " " + str(y), args))
    try:
        if silent:
            return subprocess.call(args, stdout=open(os.devnull, 'w'),
                                   stderr=open(os.devnull, 'w'))
        else:
            return subprocess.call(args)
    except IOError:
        return -1
    except OSError:
        return -1


def extract_zip(archive, destination, silent=False):
    import zipfile
    try:
        zipfile.ZipFile(archive).extractall(destination)
        return 0
    except zipfile.BadZipFile:
        if not silent:
            pass
            # logger.info("[*] could not extract zip %s via python, trying system call" % archive)

    return call_for_ret_code(['unzip', archive, '-d', destination], silent=silent)


def verify_folder(folder):
    if folder[-1] != '/':
        folder += '/'
    return folder