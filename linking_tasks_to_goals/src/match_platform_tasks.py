from .models import Task, Goal
from .convictional_tasks import get_convictional_tasks
from .platform_tasks_matching_approaches.approach_1 import approach_1
from .platform_tasks_matching_approaches.approach_2 import approach_2
from .platform_tasks_matching_approaches.approach_3 import approach_3
from .platform_tasks_matching_approaches.approach_4 import approach_4
from .platform_tasks_matching_approaches.approach_5 import approach_5
from .platform_tasks_matching_approaches.approach_6 import approach_6
from .platform_tasks_matching_approaches.approach_7 import approach_7
from .platform_tasks_matching_approaches.approach_8 import approach_8
from .platform_tasks_matching_approaches.approach_9 import approach_9
from .platform_tasks_matching_approaches.approach_10 import approach_10


async def match_platform_tasks_to_goals(goals: list[Goal]):
    """
    This part of the experiment is for matching platform tasks to goals.

    The first step is to get the tasks from the platform.
    The next step is to try and match goals to the platform tasks.
    """
    print("Matching platform tasks to goals...")

    # Get the tasks data from the platform
    tasks: list[Task] = get_convictional_tasks(load_from_cache=True)

    # Approach 1
    # await approach_1(goals, tasks, load_embeddings_from_cache=True)

    # Approach 2
    # await approach_2(goals, tasks, load_embeddings_from_cache=True)

    # Approach 3
    # await approach_3(goals, tasks, load_embeddings_from_cache=True)

    # # Approach 4
    # await approach_4(goals, tasks, load_embeddings_from_cache=True)

    # # Approach 5
    # await approach_5(goals, tasks, load_embeddings_from_cache=True)

    # # Approach 6
    # await approach_6(goals, tasks, load_embeddings_from_cache=True)

    # # Approach 7
    # await approach_7(goals, tasks, load_embeddings_from_cache=True)

    # # Approach 8
    # await approach_8(goals, tasks, load_embeddings_from_cache=True)

    # # Approach 9
    # await approach_9(goals, tasks, load_embeddings_from_cache=True)

    # Approach 10
    await approach_10(goals, tasks, load_embeddings_from_cache=True)
