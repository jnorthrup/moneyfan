#!/usr/bin/env python3
"""Conductor CLI - A tool for managing project tracks, phases, and tasks.

Usage:
    python conductor/conductor.py status
    python conductor/conductor.py next
    python conductor/conductor.py start "Task name"
    python conductor/conductor.py complete "Task name"
    python conductor/conductor.py checkpoint "Phase 1: Name"
    python conductor/conductor.py new <track_id> <description>
    python conductor/conductor.py list
    python conductor/conductor.py summary
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from track import Task, TaskStatus, Track, TrackManager

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[37m"
ANSI_DIM = "\033[2m"


def get_current_commit_sha() -> str:
    """Get current git commit SHA (7 chars)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:7]
    except subprocess.CalledProcessError:
        return ""


def attach_git_note(commit_sha: str, note: str) -> bool:
    """Attach a note to a commit."""
    try:
        subprocess.run(
            ["git", "notes", "add", "-m", note, commit_sha],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        try:
            subprocess.run(
                ["git", "notes", "append", "-m", note, commit_sha],
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False


def commit_plan_update(message: str) -> bool:
    """Commit the updated plan.md."""
    try:
        subprocess.run(
            ["git", "add", "plan.md"],
            capture_output=True,
            text=True,
            check=True
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


class Conductor:
    """Orchestrates the conductor workflow."""

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path(__file__).parent
        self.manager = TrackManager(base_path)

    def _color_status(self, status: str) -> str:
        """Colorize status string."""
        colors = {
            "new": ANSI_CYAN,
            "in_progress": ANSI_YELLOW,
            "active": ANSI_YELLOW,
            "completed": ANSI_GREEN,
            "superseded": ANSI_DIM,
        }
        color = colors.get(status, ANSI_WHITE)
        return f"{color}{status}{ANSI_RESET}"

    def _color_task_status(self, status: TaskStatus) -> str:
        """Colorize task status marker."""
        markers = {
            TaskStatus.PENDING: f"{ANSI_DIM}[ ]{ANSI_RESET}",
            TaskStatus.IN_PROGRESS: f"{ANSI_YELLOW}[~]{ANSI_RESET}",
            TaskStatus.COMPLETED: f"{ANSI_GREEN}[x]{ANSI_RESET}",
        }
        return markers.get(status, "[?]")

    def _print_header(self, text: str) -> None:
        """Print a formatted header."""
        print(f"\n{ANSI_BOLD}{ANSI_BLUE}{'=' * 60}{ANSI_RESET}")
        print(f"{ANSI_BOLD}{ANSI_BLUE}  {text}{ANSI_RESET}")
        print(f"{ANSI_BOLD}{ANSI_BLUE}{'=' * 60}{ANSI_RESET}\n")

    def _print_section(self, text: str) -> None:
        """Print a formatted section."""
        print(f"\n{ANSI_BOLD}{ANSI_CYAN}▶ {text}{ANSI_RESET}")

    def _print_success(self, text: str) -> None:
        """Print a success message."""
        print(f"{ANSI_GREEN}✓ {text}{ANSI_RESET}")

    def _print_error(self, text: str) -> None:
        """Print an error message."""
        print(f"{ANSI_RED}✗ {text}{ANSI_RESET}", file=sys.stderr)

    def _print_warning(self, text: str) -> None:
        """Print a warning message."""
        print(f"{ANSI_YELLOW}⚠ {text}{ANSI_RESET}")

    def _print_info(self, text: str) -> None:
        """Print an info message."""
        print(f"{ANSI_DIM}ℹ {text}{ANSI_RESET}")

    def run_status(self) -> int:
        """Display all tracks and their progress."""
        tracks = self.manager.list_tracks()

        if not tracks:
            self._print_info("No tracks found. Create one with: conductor new <track_id> <description>")
            return 0

        self._print_header("Track Status")

        for track in tracks:
            summary = track.get_status_summary()
            status_colored = self._color_status(track.status)
            progress_bar = self._progress_bar(summary["progress_pct"])

            print(f"{ANSI_BOLD}{track.track_id}{ANSI_RESET} {status_colored}")
            print(f"  {track.description or 'No description'}")
            print(f"  {progress_bar} {summary['progress_pct']}%")
            print(f"  Tasks: {ANSI_GREEN}{summary['completed']}{ANSI_RESET}/"
                  f"{ANSI_YELLOW}{summary['in_progress']}{ANSI_RESET}/"
                  f"{ANSI_DIM}{summary['pending']}{ANSI_RESET} "
                  f"(done/in-progress/pending)")
            print(f"  Phases: {summary['phases']}")
            print()

        active = self.manager.get_active_track()
        if active:
            self._print_info(f"Active track: {active.track_id}")

        return 0

    def _progress_bar(self, pct: float, width: int = 20) -> str:
        """Create a progress bar string."""
        filled = int(pct / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"{ANSI_GREEN}{bar}{ANSI_RESET}"

    def run_next(self) -> int:
        """Show next task and guide user to start it."""
        track = self.manager.get_active_track()

        if not track:
            self._print_error("No active track found")
            return 1

        next_task = track.get_next_task()

        if not next_task:
            self._print_success(f"All tasks complete in track: {track.track_id}")
            return 0

        self._print_header("Next Task")
        print(f"{ANSI_BOLD}Track:{ANSI_RESET} {track.track_id}")
        print(f"{ANSI_BOLD}Phase:{ANSI_RESET} {next_task.phase}")
        print(f"{ANSI_BOLD}Task:{ANSI_RESET}  {next_task.name}")
        print()

        if next_task.subtasks:
            self._print_section("Subtasks")
            for subtask in next_task.subtasks:
                print(f"  {ANSI_DIM}[ ]{ANSI_RESET} {subtask}")
            print()

        self._print_section("To start this task, run:")
        print(f"  {ANSI_CYAN}conductor start \"{next_task.name}\"{ANSI_RESET}")

        return 0

    def run_start(self, task_name: str) -> int:
        """Start a task following workflow.md."""
        track = self.manager.get_active_track()

        if not track:
            self._print_error("No active track found")
            return 1

        task = track.get_task_by_name(task_name)
        if not task:
            self._print_error(f"Task not found: {task_name}")
            return 1

        if task.status == TaskStatus.COMPLETED:
            self._print_warning(f"Task already completed: {task_name}")
            return 0

        if task.status == TaskStatus.IN_PROGRESS:
            self._print_warning(f"Task already in progress: {task_name}")
            return 0

        if not track.start_task(task_name):
            self._print_error(f"Failed to start task: {task_name}")
            return 1

        if not track.save():
            self._print_error("Failed to save track")
            return 1

        self._print_success(f"Started task: {task_name}")
        print()
        self._print_section("Task Workflow")
        print("1. Write failing tests (Red phase)")
        print("2. Implement to pass tests (Green phase)")
        print("3. Refactor if needed")
        print("4. Verify coverage (>80%)")
        print("5. Commit changes")
        print(f"6. Run: {ANSI_CYAN}conductor complete \"{task_name}\"{ANSI_RESET}")

        return 0

    def run_complete(self, task_name: str, commit_sha: Optional[str] = None) -> int:
        """Complete a task following workflow.md."""
        track = self.manager.get_active_track()

        if not track:
            self._print_error("No active track found")
            return 1

        task = track.get_task_by_name(task_name)
        if not task:
            self._print_error(f"Task not found: {task_name}")
            return 1

        if task.status == TaskStatus.COMPLETED:
            self._print_warning(f"Task already completed: {task_name}")
            return 0

        if commit_sha is None:
            commit_sha = get_current_commit_sha()
            if not commit_sha:
                self._print_error("Could not get current commit SHA")
                return 1

        if not track.complete_task(task_name, commit_sha):
            self._print_error(f"Failed to complete task: {task_name}")
            return 1

        if not track.save():
            self._print_error("Failed to save track")
            return 1

        self._print_success(f"Completed task: {task_name}")
        print(f"  Commit SHA: {ANSI_CYAN}{commit_sha}{ANSI_RESET}")

        note_content = f"""Task: {task_name}
Phase: {task.phase}
Status: Completed
Commit: {commit_sha}

Summary: Task marked as completed via conductor CLI.
"""
        if attach_git_note(commit_sha, note_content):
            self._print_success("Git note attached to commit")
        else:
            self._print_warning("Could not attach git note")

        print()
        self._print_section("Next Steps")
        print(f"1. Run: {ANSI_CYAN}conductor next{ANSI_RESET} to see next task")
        print(f"2. Or run: {ANSI_CYAN}conductor status{ANSI_RESET} to see progress")

        return 0

    def run_checkpoint(self, phase_name: str) -> int:
        """Create a phase checkpoint."""
        track = self.manager.get_active_track()

        if not track:
            self._print_error("No active track found")
            return 1

        phase_found = False
        for phase in track.phases:
            if phase.name == phase_name:
                phase_found = True
                break

        if not phase_found:
            self._print_error(f"Phase not found: {phase_name}")
            return 1

        commit_sha = get_current_commit_sha()
        if not commit_sha:
            self._print_error("Could not get current commit SHA")
            return 1

        if not track.add_phase_checkpoint(phase_name, commit_sha):
            self._print_error(f"Failed to add checkpoint to phase: {phase_name}")
            return 1

        if not track.save():
            self._print_error("Failed to save track")
            return 1

        self._print_success(f"Created checkpoint for phase: {phase_name}")
        print(f"  Checkpoint SHA: {ANSI_CYAN}{commit_sha}{ANSI_RESET}")

        note_content = f"""Checkpoint: {phase_name}
Commit: {commit_sha}

This checkpoint marks the completion of the phase verification protocol.
"""
        if attach_git_note(commit_sha, note_content):
            self._print_success("Git note attached to checkpoint commit")

        return 0

    def run_new(self, track_id: str, description: str) -> int:
        """Create a new track."""
        existing = self.manager.get_track(track_id)
        if existing:
            self._print_error(f"Track already exists: {track_id}")
            return 1

        track = self.manager.create_track(track_id, description)
        self._print_success(f"Created track: {track_id}")
        print(f"  Description: {description}")
        print(f"  Path: {track.track_path}")
        print()
        self._print_section("Next Steps")
        print(f"1. Add phases and tasks to: {track.plan_path}")
        print(f"2. Run: {ANSI_CYAN}conductor status{ANSI_RESET} to view")

        return 0

    def run_list(self) -> int:
        """List all tasks in current track."""
        track = self.manager.get_active_track()

        if not track:
            self._print_error("No active track found")
            return 1

        self._print_header(f"Tasks: {track.track_id}")

        for phase in track.phases:
            print(f"\n{ANSI_BOLD}{ANSI_CYAN}{phase.name}{ANSI_RESET}")
            if phase.checkpoint_sha:
                print(f"  {ANSI_DIM}[checkpoint: {phase.checkpoint_sha}]{ANSI_RESET}")

            for task in phase.tasks:
                marker = self._color_task_status(task.status)
                commit_part = f" {ANSI_DIM}[{task.commit_sha}]{ANSI_RESET}" if task.commit_sha else ""
                print(f"  {marker} {task.name}{commit_part}")

                for subtask in task.subtasks:
                    print(f"      {ANSI_DIM}[ ]{ANSI_RESET} {subtask}")

        print()
        return 0

    def run_summary(self) -> int:
        """Show detailed summary of current track."""
        track = self.manager.get_active_track()

        if not track:
            self._print_error("No active track found")
            return 1

        summary = track.get_status_summary()

        self._print_header(f"Track Summary: {track.track_id}")

        print(f"{ANSI_BOLD}Description:{ANSI_RESET} {track.description or 'No description'}")
        print(f"{ANSI_BOLD}Type:{ANSI_RESET}        {track.type}")
        print(f"{ANSI_BOLD}Status:{ANSI_RESET}      {self._color_status(track.status)}")
        print(f"{ANSI_BOLD}Created:{ANSI_RESET}     {track.created_at}")
        print(f"{ANSI_BOLD}Updated:{ANSI_RESET}     {track.updated_at}")
        print()

        self._print_section("Progress")
        progress_bar = self._progress_bar(summary["progress_pct"], width=30)
        print(f"  {progress_bar} {summary['progress_pct']}%")
        print()

        self._print_section("Task Counts")
        print(f"  Total:       {summary['total_tasks']}")
        print(f"  {ANSI_GREEN}Completed:{ANSI_RESET}   {summary['completed']}")
        print(f"  {ANSI_YELLOW}In Progress:{ANSI_RESET} {summary['in_progress']}")
        print(f"  {ANSI_DIM}Pending:{ANSI_RESET}     {summary['pending']}")
        print()

        self._print_section("Phases")
        print(f"  Total phases: {summary['phases']}")
        for phase in track.phases:
            completed = sum(1 for t in phase.tasks if t.status == TaskStatus.COMPLETED)
            total = len(phase.tasks)
            checkpoint = f" {ANSI_DIM}[{phase.checkpoint_sha}]{ANSI_RESET}" if phase.checkpoint_sha else ""
            print(f"    • {phase.name}: {completed}/{total}{checkpoint}")

        if track.is_complete():
            print()
            self._print_success("Track is complete!")

        return 0


def cli() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="conductor",
        description="CLI tool for managing project tracks, phases, and tasks"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("status", help="Show status of all tracks")
    subparsers.add_parser("next", help="Show the next task to work on")

    start_parser = subparsers.add_parser("start", help="Mark a task as in_progress")
    start_parser.add_argument("task_name", help="Name of the task to start")

    complete_parser = subparsers.add_parser("complete", help="Mark a task as completed")
    complete_parser.add_argument("task_name", help="Name of the task to complete")
    complete_parser.add_argument("--sha", help="Commit SHA (defaults to current)")

    checkpoint_parser = subparsers.add_parser("checkpoint", help="Add a phase checkpoint")
    checkpoint_parser.add_argument("phase_name", help="Name of the phase")

    new_parser = subparsers.add_parser("new", help="Create a new track")
    new_parser.add_argument("track_id", help="Unique track identifier")
    new_parser.add_argument("description", help="Track description")

    subparsers.add_parser("list", help="List all tasks in current track")
    subparsers.add_parser("summary", help="Show detailed summary of current track")

    args = parser.parse_args()

    conductor = Conductor()

    if args.command == "status":
        sys.exit(conductor.run_status())
    elif args.command == "next":
        sys.exit(conductor.run_next())
    elif args.command == "start":
        sys.exit(conductor.run_start(args.task_name))
    elif args.command == "complete":
        sys.exit(conductor.run_complete(args.task_name, args.sha))
    elif args.command == "checkpoint":
        sys.exit(conductor.run_checkpoint(args.phase_name))
    elif args.command == "new":
        sys.exit(conductor.run_new(args.track_id, args.description))
    elif args.command == "list":
        sys.exit(conductor.run_list())
    elif args.command == "summary":
        sys.exit(conductor.run_summary())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
