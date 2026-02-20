"""Track management module for conductor system.

Manages project tracks, phases, and tasks for structured development workflows.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class Task:
    """A single unit of work within a phase."""
    name: str
    status: TaskStatus
    phase: str
    commit_sha: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)

    def to_markdown(self, indent: int = 0) -> str:
        """Convert task to markdown format."""
        prefix = "  " * indent
        status_marker = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.IN_PROGRESS: "[~]",
            TaskStatus.COMPLETED: "[x]",
        }[self.status]
        
        commit_part = f" [{self.commit_sha}]" if self.commit_sha else ""
        line = f"{prefix}- {status_marker} Task: {self.name}{commit_part}"
        
        for subtask in self.subtasks:
            line += f"\n{prefix}    - [ ] {subtask}"
        
        return line


@dataclass
class Phase:
    """A group of related tasks within a track."""
    name: str
    tasks: List[Task] = field(default_factory=list)
    checkpoint_sha: Optional[str] = None

    def to_markdown(self) -> str:
        """Convert phase to markdown format."""
        lines = [f"## {self.name}", ""]
        for task in self.tasks:
            lines.append(task.to_markdown(indent=0))
        return "\n".join(lines)


class Track:
    """A major work stream containing phases and tasks."""

    def __init__(self, track_id: str, base_path: Path):
        """Initialize a track.

        Args:
            track_id: Unique identifier for the track.
            base_path: Root path for conductor tracks directory.
        """
        self.track_id = track_id
        self.base_path = base_path
        self.track_path = base_path / track_id
        
        self.phases: List[Phase] = []
        self.type: str = "feature"
        self.status: str = "new"
        self.created_at: str = datetime.utcnow().isoformat() + "Z"
        self.updated_at: str = self.created_at
        self.description: str = ""
        self._loaded: bool = False

    @property
    def plan_path(self) -> Path:
        """Path to the plan.md file."""
        return self.track_path / "plan.md"

    @property
    def metadata_path(self) -> Path:
        """Path to the metadata.json file."""
        return self.track_path / "metadata.json"

    def load(self) -> bool:
        """Load track from plan.md and metadata.json.

        Returns:
            True if loading succeeded, False otherwise.
        """
        if not self.track_path.exists():
            return False

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    metadata = json.load(f)
                self.type = metadata.get("type", "feature")
                self.status = metadata.get("status", "new")
                self.created_at = metadata.get("created_at", self.created_at)
                self.updated_at = metadata.get("updated_at", self.updated_at)
                self.description = metadata.get("description", "")
            except (json.JSONDecodeError, IOError):
                pass

        if self.plan_path.exists():
            try:
                with open(self.plan_path, "r") as f:
                    content = f.read()
                self._parse_plan(content)
            except IOError:
                pass

        self._loaded = True
        return True

    def _parse_plan(self, content: str) -> None:
        """Parse plan.md content into phases and tasks."""
        self.phases = []
        
        phase_pattern = re.compile(r"^## (Phase \d+?: .+?)$", re.MULTILINE)
        task_pattern = re.compile(
            r"^- \[([ x~])\] Task: (.+?)(?:\s+\[([a-f0-9]+)\])?$",
            re.MULTILINE
        )
        subtask_pattern = re.compile(r"^    - \[ \] (.+)$", re.MULTILINE)

        phase_matches = list(phase_pattern.finditer(content))
        
        for i, phase_match in enumerate(phase_matches):
            phase_name = phase_match.group(1)
            start_pos = phase_match.end()
            end_pos = phase_matches[i + 1].start() if i + 1 < len(phase_matches) else len(content)
            phase_content = content[start_pos:end_pos]
            
            phase = Phase(name=phase_name)
            
            task_matches = list(task_pattern.finditer(phase_content))
            for task_match in task_matches:
                status_marker = task_match.group(1)
                task_name = task_match.group(2).strip()
                commit_sha = task_match.group(3)
                
                status = {
                    " ": TaskStatus.PENDING,
                    "~": TaskStatus.IN_PROGRESS,
                    "x": TaskStatus.COMPLETED,
                }[status_marker]
                
                task = Task(
                    name=task_name,
                    status=status,
                    phase=phase_name,
                    commit_sha=commit_sha,
                )
                
                task_end = task_match.end()
                next_task_start = task_matches[task_matches.index(task_match) + 1].start() if task_matches.index(task_match) + 1 < len(task_matches) else len(phase_content)
                task_section = phase_content[task_end:next_task_start]
                
                subtask_matches = subtask_pattern.findall(task_section)
                task.subtasks = [s.strip() for s in subtask_matches]
                
                phase.tasks.append(task)
            
            self.phases.append(phase)

    def save(self) -> bool:
        """Save track state to plan.md and metadata.json.

        Returns:
            True if saving succeeded, False otherwise.
        """
        try:
            self.track_path.mkdir(parents=True, exist_ok=True)
            
            self.updated_at = datetime.utcnow().isoformat() + "Z"
            
            metadata = {
                "track_id": self.track_id,
                "type": self.type,
                "status": self.status,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "description": self.description,
            }
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            plan_content = f"# Plan: {self.description or self.track_id}\n\n"
            plan_content += f"Track ID: `{self.track_id}`\n\n---\n\n"
            
            for phase in self.phases:
                plan_content += phase.to_markdown() + "\n\n---\n\n"
            
            with open(self.plan_path, "w") as f:
                f.write(plan_content.rstrip() + "\n")
            
            return True
        except (IOError, OSError):
            return False

    def get_next_task(self) -> Optional[Task]:
        """Get the next pending task.

        Returns:
            The next pending task, or None if all tasks are complete.
        """
        for phase in self.phases:
            for task in phase.tasks:
                if task.status == TaskStatus.PENDING:
                    return task
        return None

    def start_task(self, task_name: str) -> bool:
        """Mark a task as in progress.

        Args:
            task_name: Name of the task to start.

        Returns:
            True if the task was found and updated, False otherwise.
        """
        for phase in self.phases:
            for task in phase.tasks:
                if task.name == task_name:
                    task.status = TaskStatus.IN_PROGRESS
                    return True
        return False

    def complete_task(self, task_name: str, commit_sha: str) -> bool:
        """Mark a task as completed.

        Args:
            task_name: Name of the task to complete.
            commit_sha: Git commit SHA associated with the completion.

        Returns:
            True if the task was found and updated, False otherwise.
        """
        for phase in self.phases:
            for task in phase.tasks:
                if task.name == task_name:
                    task.status = TaskStatus.COMPLETED
                    task.commit_sha = commit_sha
                    return True
        return False

    def add_phase_checkpoint(self, phase_name: str, commit_sha: str) -> bool:
        """Add a checkpoint commit to a phase.

        Args:
            phase_name: Name of the phase.
            commit_sha: Git commit SHA for the checkpoint.

        Returns:
            True if the phase was found and updated, False otherwise.
        """
        for phase in self.phases:
            if phase.name == phase_name:
                phase.checkpoint_sha = commit_sha
                return True
        return False

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of track progress.

        Returns:
            Dictionary with status counts and progress percentage.
        """
        total_tasks = 0
        pending = 0
        in_progress = 0
        completed = 0

        for phase in self.phases:
            for task in phase.tasks:
                total_tasks += 1
                if task.status == TaskStatus.PENDING:
                    pending += 1
                elif task.status == TaskStatus.IN_PROGRESS:
                    in_progress += 1
                elif task.status == TaskStatus.COMPLETED:
                    completed += 1

        progress_pct = (completed / total_tasks * 100) if total_tasks > 0 else 0.0

        return {
            "track_id": self.track_id,
            "status": self.status,
            "type": self.type,
            "total_tasks": total_tasks,
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "progress_pct": round(progress_pct, 1),
            "phases": len(self.phases),
        }

    def get_task_by_name(self, task_name: str) -> Optional[Task]:
        """Get a task by name.

        Args:
            task_name: Name of the task to find.

        Returns:
            The task if found, None otherwise.
        """
        for phase in self.phases:
            for task in phase.tasks:
                if task.name == task_name:
                    return task
        return None

    def is_complete(self) -> bool:
        """Check if all tasks in the track are completed.

        Returns:
            True if all tasks are completed, False otherwise.
        """
        for phase in self.phases:
            for task in phase.tasks:
                if task.status != TaskStatus.COMPLETED:
                    return False
        return True


class TrackManager:
    """Manages multiple tracks in a conductor system."""

    def __init__(self, base_path: Path):
        """Initialize the track manager.

        Args:
            base_path: Root path for conductor tracks directory.
        """
        self.base_path = base_path
        self.tracks_path = base_path / "tracks"

    def list_tracks(self) -> List[Track]:
        """List all tracks.

        Returns:
            List of all Track objects.
        """
        tracks: List[Track] = []
        
        if not self.tracks_path.exists():
            return tracks

        for track_dir in sorted(self.tracks_path.iterdir()):
            if track_dir.is_dir() and (track_dir / "metadata.json").exists():
                track = Track(track_dir.name, self.tracks_path)
                track.load()
                tracks.append(track)

        return tracks

    def get_active_track(self) -> Optional[Track]:
        """Get the first non-completed track.

        Returns:
            The first track with status != 'completed', or None.
        """
        for track in self.list_tracks():
            if track.status not in ("completed", "superseded"):
                return track
        return None

    def create_track(
        self,
        track_id: str,
        description: str,
        track_type: str = "feature"
    ) -> Track:
        """Create a new track.

        Args:
            track_id: Unique identifier for the track.
            description: Description of the track.
            track_type: Type of track (feature, bugfix, refactor, etc.).

        Returns:
            The newly created Track object.
        """
        track = Track(track_id, self.tracks_path)
        track.description = description
        track.type = track_type
        track.status = "new"
        track.save()
        return track

    def get_track(self, track_id: str) -> Optional[Track]:
        """Get a specific track by ID.

        Args:
            track_id: The track identifier.

        Returns:
            The Track if found, None otherwise.
        """
        track = Track(track_id, self.tracks_path)
        if track.load():
            return track
        return None

    def archive_track(self, track_id: str) -> bool:
        """Archive a track by marking it as superseded.

        Args:
            track_id: The track identifier.

        Returns:
            True if successful, False otherwise.
        """
        track = self.get_track(track_id)
        if track:
            track.status = "superseded"
            return track.save()
        return False
