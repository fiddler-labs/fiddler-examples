"""Safe iteration utilities for Fiddler objects.

This module provides utilities for safely iterating over Fiddler projects,
models, and other objects with built-in error handling.
"""

from typing import Iterator, Optional, List, Tuple, Dict
import logging

try:
    import fiddler as fdl
except ImportError:
    raise ImportError(
        'fiddler-client is required. Install it with: pip install fiddler-client'
    )

logger = logging.getLogger(__name__)


def iterate_projects_safe(
    names: Optional[List[str]] = None,
    on_error: str = 'warn'
) -> Iterator[Tuple[fdl.Project, Optional[Exception]]]:
    """Safely iterate over projects with error handling.

    Args:
        names: Optional list of project names to filter
        on_error: Error handling strategy:
            - 'warn': Log warning and continue (default)
            - 'skip': Silently skip and continue
            - 'raise': Raise the exception

    Yields:
        Tuple of (project, error) where error is None on success

    Example:
        ```python
        from fiddler_utils import iterate_projects_safe

        for project, error in iterate_projects_safe():
            if error:
                print(f"Error: {error}")
                continue

            print(f"Project: {project.name}")
        ```
    """
    try:
        projects = list(fdl.Project.list())

        if names:
            projects = [p for p in projects if p.name in names]

        for project in projects:
            try:
                # Verify project is accessible
                yield (project, None)
            except Exception as e:
                if on_error == 'raise':
                    raise
                elif on_error == 'warn':
                    logger.warning(f'Error accessing project {project.name}: {e}')

                yield (project, e)

    except Exception as e:
        if on_error == 'raise':
            raise
        elif on_error == 'warn':
            logger.error(f'Failed to list projects: {e}')


def iterate_models_safe(
    project_ids: Optional[List[str]] = None,
    fetch_full: bool = True,
    on_error: str = 'warn'
) -> Iterator[Tuple[fdl.Project, fdl.Model, Optional[Exception]]]:
    """Safely iterate over all models with error handling.

    This utility handles common iteration patterns:
    - Traversing all projects
    - Listing models in each project
    - Optionally fetching full model details
    - Graceful error handling at each level

    Args:
        project_ids: Optional list of project IDs to filter
        fetch_full: If True, fetch full Model (not ModelCompact)
        on_error: Error handling strategy:
            - 'warn': Log warning and continue (default)
            - 'skip': Silently skip and continue
            - 'raise': Raise the exception

    Yields:
        Tuple of (project, model, error) where error is None on success

    Example:
        ```python
        from fiddler_utils import iterate_models_safe

        for project, model, error in iterate_models_safe(fetch_full=True):
            if error:
                print(f"Failed to fetch {project.name}/{model.name}: {error}")
                continue

            # Process model
            print(f"Processing {model.name} with {len(model.spec.inputs or [])} features")
        ```

    Example with specific projects:
        ```python
        project_ids = ['proj-uuid-1', 'proj-uuid-2']

        for project, model, error in iterate_models_safe(
            project_ids=project_ids,
            fetch_full=True,
            on_error='skip'  # Silently skip errors
        ):
            if error:
                continue

            # Only successful fetches reach here
            print(f"{project.name}/{model.name}")
        ```
    """
    try:
        # List projects
        projects = list(fdl.Project.list())

        if project_ids:
            projects = [p for p in projects if str(p.id) in project_ids]

        for project in projects:
            try:
                # List models in project
                models = list(fdl.Model.list(project_id=project.id))

                for model_compact in models:
                    try:
                        if fetch_full:
                            # Fetch full model details
                            model = fdl.Model.get(id_=model_compact.id)
                        else:
                            model = model_compact

                        yield (project, model, None)

                    except Exception as e:
                        if on_error == 'raise':
                            raise
                        elif on_error == 'warn':
                            logger.warning(
                                f'Error fetching model {model_compact.name} '
                                f'in project {project.name}: {e}'
                            )

                        # Yield with error
                        yield (project, model_compact, e)

            except Exception as e:
                if on_error == 'raise':
                    raise
                elif on_error == 'warn':
                    logger.warning(f'Error listing models for project {project.name}: {e}')

    except Exception as e:
        if on_error == 'raise':
            raise
        elif on_error == 'warn':
            logger.error(f'Failed to list projects: {e}')


def count_models_by_project(
    project_ids: Optional[List[str]] = None,
    on_error: str = 'warn'
) -> Dict[str, int]:
    """Count models in each project.

    Args:
        project_ids: Optional list of project IDs to filter
        on_error: Error handling strategy

    Returns:
        Dictionary mapping project name to model count

    Example:
        ```python
        from fiddler_utils.iteration import count_models_by_project

        counts = count_models_by_project()
        for project_name, count in sorted(counts.items()):
            print(f"{project_name}: {count} models")
        ```
    """
    counts = {}

    for project, model, error in iterate_models_safe(
        project_ids=project_ids,
        fetch_full=False,  # Don't need full model details
        on_error=on_error
    ):
        if error:
            continue

        if project.name not in counts:
            counts[project.name] = 0
        counts[project.name] += 1

    return counts
