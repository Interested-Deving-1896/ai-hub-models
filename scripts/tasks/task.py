# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import subprocess
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Callable
from sys import platform

from .constants import REPO_ROOT
from .github import end_group, start_group
from .test_utils import TestStatus, write_junit_testcase
from .util import BASH_EXECUTABLE, debug_mode, default_parallelism, echo, have_root


class Task(ABC):
    def __init__(
        self,
        group_name: str | None,
        raise_on_failure: bool = True,
        junit_xml_path: str | None = None,
        junit_testsuite: str = "",
        junit_name: str = "",
        junit_classname: str = "",
        prereqs: list[Task] | None = None,
        group_logs_on_ci: bool = True,
    ) -> None:
        self.group_name = group_name
        self.group_logs_on_ci = group_logs_on_ci
        self.raise_on_failure = raise_on_failure
        self.last_result: bool | None = None
        self.last_result_exception: Exception | None = None
        self.junit_xml_path = junit_xml_path
        self.junit_testsuite = junit_testsuite
        self.junit_name = junit_name
        self.junit_classname = junit_classname
        self.prereqs = prereqs or []

    @abstractmethod
    def does_work(self) -> bool:
        """Return True if this task actually does something (e.g., runs commands)."""

    @abstractmethod
    def run_task(self) -> bool:
        """Entry point for implementations: perform the task's action."""

    def _write_junit(
        self,
        status: TestStatus,
        message: str = "",
        text: str = "",
    ) -> None:
        if not self.junit_xml_path:
            return
        write_junit_testcase(
            junit_xml_path=self.junit_xml_path,
            testsuite=self.junit_testsuite,
            name=self.junit_name,
            classname=self.junit_classname,
            status=status,
            message=message,
            text=text,
        )

    def _failed_prereqs(self) -> list[Task]:
        return [p for p in self.prereqs if p.last_result is False]

    def run(self) -> bool:
        """Entry point for callers: perform any startup/teardown tasks and call run_task."""
        self.last_result_exception = None

        if self.group_name and self.group_logs_on_ci:
            start_group(self.group_name)

        failed = self._failed_prereqs()
        if failed:
            names = ", ".join(p.group_name or "unnamed" for p in failed)
            echo(f"Skipping {self.group_name}: prerequisite(s) failed ({names})")
            self._write_junit(
                TestStatus.SKIPPED,
                message=f"Skipped: prerequisite(s) failed ({names})",
            )
            self.last_result = False
            if self.group_name and self.group_logs_on_ci:
                end_group()
            return False

        try:
            result = self.run_task()
        except Exception as err:
            self.last_result_exception = err
            result = False

        if self.group_name and self.group_logs_on_ci:
            end_group()

        if not result:
            self._write_junit(
                TestStatus.FAILED,
                message=f"{self.group_name} failed. Check the logs for details.",
            )
        else:
            self._write_junit(TestStatus.PASSED)

        self.last_result = result
        if not result and self.raise_on_failure:
            raise self.last_result_exception or Exception(self.get_status_message())

        return result

    def get_status_message(self) -> str:
        if self.last_result is not None:
            if self.last_result:
                return f"{self.group_name} succeeded."
            return f"{self.group_name} failed{': ' + str(self.last_result_exception) if self.last_result_exception else ''}."
        return f"{self.group_name} has not run."


class ListTasksTask(Task):
    def __init__(self, tasks: list[str]) -> None:
        super().__init__(group_name=None)
        self.tasks = tasks

    def does_work(self) -> bool:
        return False

    def run_task(self) -> bool:
        from . import plan

        for task_name in sorted(self.tasks):
            print(task_name)
            description = plan.TASK_DESCRIPTIONS.get(task_name, None)
            if description:
                print(f"    {description}")

        return True


class NoOpTask(Task):
    """A Task that does nothing."""

    def __init__(self, group_name: str | None = None) -> None:
        super().__init__(group_name=group_name)

    def does_work(self) -> bool:
        return False

    def run_task(self) -> bool:
        return True


class RunCommandsTask(Task):
    """A Task that runs a list of commands using the shell."""

    def __init__(
        self,
        group_name: str | None,
        commands: list[str] | str,
        as_root: bool = False,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
        junit_xml_path: str | None = None,
        junit_testsuite: str = "",
        junit_name: str = "",
        junit_classname: str = "",
        prereqs: list[Task] | None = None,
        retries: int = 0,
        retry_sleep_s: float = 30.0,
    ) -> None:
        if ignore_return_codes is None:
            ignore_return_codes = []
        super().__init__(
            group_name,
            raise_on_failure,
            junit_xml_path=junit_xml_path,
            junit_testsuite=junit_testsuite,
            junit_name=junit_name,
            junit_classname=junit_classname,
            prereqs=prereqs,
        )
        if isinstance(commands, str):
            self.commands = [commands]
        else:
            self.commands = commands

        if as_root and not have_root():
            self.commands = [f"sudo {c}" for c in commands]

        self.cwd = cwd
        self.env = env
        self.ignore_return_codes = ignore_return_codes
        self.retries = retries
        self.retry_sleep_s = retry_sleep_s

    def does_work(self) -> bool:
        return True

    def run_task(self) -> bool:
        result: bool = True
        for command in self.commands:
            result = result and self._run_command(command)
            if not result:
                break
        return result

    def _run_command(self, command: str) -> bool:
        echo(f"bnt $ {command}")
        max_attempts = self.retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    cwd=self.cwd,
                    env=self.env,
                    executable=BASH_EXECUTABLE,
                )
                return True
            except subprocess.CalledProcessError as e:
                if e.returncode in self.ignore_return_codes:
                    return True
                if attempt < max_attempts:
                    echo(
                        f"Command failed (exit {e.returncode}); "
                        f"retrying in {self.retry_sleep_s}s "
                        f"(attempt {attempt}/{max_attempts})"
                    )
                    time.sleep(self.retry_sleep_s)
                    continue
                if self.raise_on_failure:
                    raise
                return False
        return True


class RunCommandsWithVenvTask(RunCommandsTask):
    """
    A Task that runs a list of commands using the shell with a specific Python
    virtual environment enabled.
    """

    def __init__(
        self,
        group_name: str | None,
        venv: str | None,
        commands: list[str] | str,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
        junit_xml_path: str | None = None,
        junit_testsuite: str = "",
        junit_name: str = "",
        junit_classname: str = "",
        prereqs: list[Task] | None = None,
        retries: int = 0,
        retry_sleep_s: float = 30.0,
    ) -> None:
        if ignore_return_codes is None:
            ignore_return_codes = []
        super().__init__(
            group_name,
            commands,
            env=env,
            raise_on_failure=raise_on_failure,
            ignore_return_codes=ignore_return_codes,
            junit_xml_path=junit_xml_path,
            junit_testsuite=junit_testsuite,
            junit_name=junit_name,
            junit_classname=junit_classname,
            prereqs=prereqs,
            retries=retries,
            retry_sleep_s=retry_sleep_s,
        )
        self.venv = venv
        self.commands = [commands] if isinstance(commands, str) else commands
        if debug_mode() and platform in ["linux", "linux2"]:
            self.commands.insert(0, "free")
        if debug_mode() and platform in ["darwin", "linux", "linux2"]:
            self.commands.insert(0, "df -h")
        if self.venv is not None:
            self.commands = [
                f"source {self.venv}/bin/activate && {command}"
                for command in self.commands
            ]


class PyTestTask(RunCommandsWithVenvTask):
    """A task to run pytest"""

    def __init__(
        self,
        group_name: str | None,
        venv: str | None,
        files_or_dirs: str,
        ignore: str | list[str] | None = None,
        parallel: bool | int | None = None,
        extra_args: str | None = None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        # Pytest returns code 5 if no tests were run. Set this to true
        # to ignore that return code (count it as "passed")
        ignore_no_tests_return_code: bool = False,
        include_pytest_cmd_in_status_message: bool = True,
        junit_xml_path: str | None = None,
        # Classname to record on a synthetic junit entry when pytest
        # itself failed without writing a testcase (e.g. conftest
        # ImportError). Not needed if pytest wrote proper entries.
        junit_classname: str = "",
        config_file: str | os.PathLike = os.path.join(
            REPO_ROOT, "src", "pyproject.toml"
        ),
        prereqs: list[Task] | None = None,
    ) -> None:
        pytest_options = f"--config-file={config_file}"

        if ignore:
            if isinstance(ignore, str):
                ignore = [ignore]
            ignores = [f"--ignore={i}" for i in ignore]
            pytest_options += f" {' '.join(ignores)}"

        if parallel:
            if isinstance(parallel, bool):
                parallel = default_parallelism()
            pytest_options += f" -n {parallel}"
            # Don't run tests that don't support parallelism
            pytest_options += ' -m "not serial"'

        pytest_options += " -ra -vvv --tb=short"

        # Add JUnit XML output option if specified
        if junit_xml_path:
            pytest_options += f" --junit-xml={junit_xml_path}"

        if extra_args:
            pytest_options += f" {extra_args}"

        pytest_options += f" {files_or_dirs}"

        default_options = "-rxXs -p no:warnings --durations-min=0.5 --durations=20"
        if debug_mode():
            if platform in ["linux", "linux2"]:
                pytest = "/usr/bin/time -v pytest"
            elif platform == "darwin":
                pytest = "/usr/bin/time pytest"
            else:
                pytest = "pytest"
        else:
            pytest = "pytest"
        command = f"{pytest} {default_options} {pytest_options} "

        self.include_pytest_cmd_in_status_message = include_pytest_cmd_in_status_message
        self._pytest_junit_xml_path = junit_xml_path
        self._pytest_junit_classname = junit_classname
        super().__init__(
            group_name,
            venv,
            command,
            env,
            raise_on_failure,
            ignore_return_codes=[5] if ignore_no_tests_return_code else [],
            prereqs=prereqs,
        )

    def get_status_message(self) -> str:
        if not self.include_pytest_cmd_in_status_message and self.last_result is False:
            return f"{self.group_name} failed."
        return super().get_status_message()

    def run(self) -> bool:
        result = super().run()
        if not result:
            self._ensure_failure_recorded_in_junit()
        return result

    def _ensure_failure_recorded_in_junit(self) -> None:
        """Write a synthetic failure entry when pytest didn't record one.

        Pytest aborts at conftest ImportError before emitting a testcase, so
        --junit-xml stays empty of failures. Downstream summary generators
        skip empty testcases and drop the model from the failure notification.
        """
        if not self._pytest_junit_xml_path or not self._pytest_junit_classname:
            return
        # If pytest recorded any failure/error, our composite failure is
        # already visible — don't add noise.
        if os.path.exists(self._pytest_junit_xml_path):
            try:
                tree = ET.parse(self._pytest_junit_xml_path)
                for tc in tree.getroot().iter("testcase"):
                    if tc.find("failure") is not None or tc.find("error") is not None:
                        return
            except ET.ParseError:
                pass
        write_junit_testcase(
            junit_xml_path=self._pytest_junit_xml_path,
            testsuite="pytest",
            name="pytest_collection",
            classname=self._pytest_junit_classname,
            status=TestStatus.ERROR,
            message=(
                "pytest failed without recording a testcase-level failure "
                "(commonly a conftest ImportError). Check the workflow log "
                f"under group '{self.group_name}' for the underlying error."
            ),
        )


class CompositeTask(Task):
    """A Task composed of a list of other Tasks."""

    def __init__(
        self,
        group_name: str | None,
        tasks: list[Task],
        continue_after_single_task_failure: bool = False,
        raise_on_failure: bool = True,
        show_subtasks_in_failure_message: bool = True,
        junit_xml_path: str | None = None,
        junit_testsuite: str = "",
        junit_name: str = "",
        junit_classname: str = "",
    ) -> None:
        super().__init__(
            group_name,
            raise_on_failure,
            junit_xml_path=junit_xml_path,
            junit_testsuite=junit_testsuite,
            junit_name=junit_name,
            junit_classname=junit_classname,
            # Don't group logs if subtasks use groups. GH can't handle nested groups.
            group_logs_on_ci=not any(t.group_logs_on_ci for t in tasks),
        )
        self.tasks = tasks
        self.continue_after_single_task_failure = continue_after_single_task_failure
        self.show_subtasks_in_failure_message = show_subtasks_in_failure_message

    def does_work(self) -> bool:
        return any(t.does_work() for t in self.tasks)

    def run_task(self) -> bool:
        self.prev_run_status = {}
        result: bool = True
        for task in self.tasks:
            try:
                task_result = task.run()
            except Exception:
                task_result = False
            result = result and task_result
            if not result and not self.continue_after_single_task_failure:
                break
        return result

    def get_status_message(self) -> str:
        if self.last_result is not None:
            if self.last_result:
                return f"{self.group_name} succeeded."
            if self.group_name is None:
                all_res = []
                for task in self.tasks:
                    if not task.last_result:
                        all_res.append(task.get_status_message())
                return "\n".join(all_res)
            res = f"{self.group_name} failed."
            if self.show_subtasks_in_failure_message:
                res += " Composite failure summary:"
                for task in self.tasks:
                    if not task.last_result:
                        res += "\n    " + task.get_status_message().replace(
                            "\n", "\n    "
                        )
            return res
        return f"{self.group_name} has not run."


class ConditionalTask(Task):
    """
    A Task that runs one of two alternatives, depending on the result of
    a predicate function call.
    """

    def __init__(
        self,
        group_name: str | None,
        condition: Callable[[], bool],
        true_task: Task,
        false_task: Task,
        raise_on_failure: bool = True,
    ) -> None:
        super().__init__(group_name, raise_on_failure)
        self.condition = condition
        self.true_task = true_task
        self.false_task = false_task

    def does_work(self) -> bool:
        if self.condition():
            return self.true_task.does_work()
        return self.false_task.does_work()

    def run_task(self) -> bool:
        if self.condition():
            return self.true_task.run()
        return self.false_task.run()
