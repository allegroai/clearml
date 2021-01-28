"""
Locking constants

Lock types:

- `LOCK_EX` exclusive lock
- `LOCK_SH` shared lock

Lock flags:

- `LOCK_NB` non-blocking

Manually unlock, only needed internally

- `LOCK_UN` unlock
"""
import os

# The actual tests will execute the code anyhow so the following code can
# safely be ignored from the coverage tests
if os.name == 'nt':  # pragma: no cover
    import msvcrt

    LOCK_EX = 0x1  #: exclusive lock
    LOCK_SH = 0x2  #: shared lock
    LOCK_NB = 0x4  #: non-blocking
    LOCK_UN = msvcrt.LK_UNLCK  #: unlock

    LOCKFILE_FAIL_IMMEDIATELY = 1
    LOCKFILE_EXCLUSIVE_LOCK = 2

elif os.name == 'posix':  # pragma: no cover
    import fcntl

    LOCK_EX = fcntl.LOCK_EX  #: exclusive lock
    LOCK_SH = fcntl.LOCK_SH  #: shared lock
    LOCK_NB = fcntl.LOCK_NB  #: non-blocking
    LOCK_UN = fcntl.LOCK_UN  #: unlock

else:  # pragma: no cover
    raise RuntimeError('PortaLocker only defined for nt and posix platforms')
