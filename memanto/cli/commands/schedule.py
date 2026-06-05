"""
MEMANTO CLI - Schedule commands (enable, disable, status).
"""

from rich.panel import Panel

from memanto.cli.commands._shared import (
    SUCCESS,
    WARNING,
    _error,
    config_manager,
    console,
    schedule_app,
)
from memanto.cli.schedule_manager import ScheduleManager


@schedule_app.command("enable")
def schedule_enable():
    """Enable daily conflict detection at configured time."""

    manager = ScheduleManager()

    # Read configured time, default to 23:55
    configured_time = config_manager.get_schedule_time()
    result = manager.enable(configured_time)

    if result.get("status") == "success":
        console.print(f"[green]OK {result.get('message')}[/green]")
    else:
        _error(result.get("message"))


@schedule_app.command("disable")
def schedule_disable():
    """Disable auto conflict detection."""

    manager = ScheduleManager()
    result = manager.disable()

    if result.get("status") == "success":
        console.print(f"[green]OK {result.get('message')}[/green]")
    else:
        _error(result.get("message"))


@schedule_app.command("status")
def schedule_status():
    """Check the status of the conflict detection schedule."""

    manager = ScheduleManager()
    result = manager.get_status()
    configured_time = config_manager.get_schedule_time()

    if result.get("enabled"):
        console.print(
            Panel(
                f"[green]Conflict Detection Automation: ENABLED[/green]\n\n"
                f"[dim]Time: {configured_time} local time daily[/dim]\n"
                f"[dim]Command: {manager.python_exe} ... "
                f"{manager.SCHEDULED_COMMAND}[/dim]",
                title="Schedule Status",
                border_style=SUCCESS,
            )
        )
    else:
        console.print(
            Panel(
                "[yellow]Conflict Detection Automation: DISABLED[/yellow]\n\n"
                "[dim]Run 'memanto schedule enable' to activate automatic "
                "conflict detection.[/dim]",
                title="Schedule Status",
                border_style=WARNING,
            )
        )
