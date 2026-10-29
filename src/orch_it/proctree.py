"""One process and every process it starts, started and stopped the same way on every OS (#396
decision 1: "process start and stop go through one helper per OS"; first built for #394 decision 2).

On Windows the process starts with no console window and suspended, joins a job object that
kills its members when closed, and is then resumed, so no child escapes the job; when Windows
will not make a job, `taskkill /T /F` stops the tree instead. Elsewhere the process leads its own
session and process group, and `kill` signals the group (a process that starts its own session
escapes it). `kill` waits, bounded, for the process to end and is safe to call twice.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

KILL_WAIT_SECONDS = 30.0


class ProcessTree:
    """A started command and every process it starts. `stdout` and `stderr` are what `Popen`
    takes (a pipe, a file object, DEVNULL); stdin is always DEVNULL."""

    def __init__(self, process: subprocess.Popen, job=None):
        self.process = process
        self.job = job

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    def poll(self) -> int | None:
        """The exit code of the process, or None while it runs."""
        return self.process.poll()

    @classmethod
    def start(cls, argv, *, cwd, env, stdout, stderr) -> ProcessTree:
        common = {"cwd": cwd, "env": env, "stdin": subprocess.DEVNULL, "stdout": stdout,
                  "stderr": stderr}
        if sys.platform != "win32":
            return cls(subprocess.Popen(argv, start_new_session=True, **common))
        process = subprocess.Popen(argv, creationflags=_win.CREATE_SUSPENDED | _win.CREATE_NO_WINDOW,
                                   **common)
        try:
            job = _win.job_for(process)
            _win.resume(process)
        except BaseException:  # never leave a suspended process nobody kills
            process.kill()
            process.wait(timeout=KILL_WAIT_SECONDS)
            raise
        return cls(process, job)

    def kill(self) -> None:
        if sys.platform == "win32":
            job, self.job = self.job, None  # a closed job handle is never used twice
            _win.kill(self.process, job)
        else:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            self.process.wait(timeout=KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=KILL_WAIT_SECONDS)


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _Win:
        CREATE_SUSPENDED = 0x00000004
        CREATE_NO_WINDOW = 0x08000000
        KILL_ON_JOB_CLOSE = 0x00002000
        EXTENDED_LIMIT_INFORMATION = 9

        class _Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class _Extended(ctypes.Structure):
            pass

        _Extended._fields_ = [("BasicLimitInformation", _Basic),
                              ("IoInfo", ctypes.c_ulonglong * 6),
                              ("ProcessMemoryLimit", ctypes.c_size_t),
                              ("JobMemoryLimit", ctypes.c_size_t),
                              ("PeakProcessMemoryUsed", ctypes.c_size_t),
                              ("PeakJobMemoryUsed", ctypes.c_size_t)]

        def __init__(self):
            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self.ntdll = ctypes.WinDLL("ntdll")
            k = self.kernel32
            k.CreateJobObjectW.restype = wintypes.HANDLE
            k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                  wintypes.DWORD]
            k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k.CloseHandle.argtypes = [wintypes.HANDLE]
            self.ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]

        def job_for(self, process: subprocess.Popen):
            """A job object holding `process` (still suspended), or None when Windows would not
            make one; then `kill` falls back to `taskkill /T`."""

            k = self.kernel32
            job = k.CreateJobObjectW(None, None)
            if not job:
                return None
            info = self._Extended()
            info.BasicLimitInformation.LimitFlags = self.KILL_ON_JOB_CLOSE
            if not (k.SetInformationJobObject(job, self.EXTENDED_LIMIT_INFORMATION,
                                              ctypes.byref(info), ctypes.sizeof(info))
                    and k.AssignProcessToJobObject(job, int(process._handle))):
                k.CloseHandle(job)
                return None
            return job

        def resume(self, process: subprocess.Popen) -> None:
            self.ntdll.NtResumeProcess(int(process._handle))

        def kill(self, process: subprocess.Popen, job) -> None:
            if job is not None:
                self.kernel32.TerminateJobObject(job, 1)
                self.kernel32.CloseHandle(job)
                return
            if process.poll() is not None:  # taskkill walks from a live parent; its pid may be reused
                return
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                           capture_output=True, timeout=KILL_WAIT_SECONDS,
                           creationflags=self.CREATE_NO_WINDOW)

    _win = _Win()
