import time
from typing import List, Generator


def chunks(l, n):
    # type: (list[str],int) -> Generator[List[(str,str)]]
    """
    <Description>

    :param list[str] l:
    :param int n:
    :return:
    """
    n = max(1, n)
    return (l[i : i + n] for i in range(0, len(l), n))


class Timer:
    def __init__(self):
        self.dict_start = {
            "read_general_information": {},
            "read_artifacts": {},
            "read_metrics": {},
            "read_params": {},
            "read_tags": {},
            "transmit_information": {},
            "transmit_metrics": {},
            "transmit_artifacts": {},
            "Task.create": {},
            "Task.get_task": {},
            "task.export_task": {},
            "task.update_task": {},
            "task.connect_configuration": {},
        }
        self.dict_end = {
            "read_general_information": {},
            "read_artifacts": {},
            "read_metrics": {},
            "read_params": {},
            "read_tags": {},
            "transmit_information": {},
            "transmit_metrics": {},
            "transmit_artifacts": {},
            "Task.create": {},
            "Task.get_task": {},
            "task.export_task": {},
            "task.update_task": {},
            "task.connect_configuration": {},
        }
        self.current_milli_time = lambda: int(round(time.time() * 1000))

    def start(self, operation, thread_id, experiment_id):
        self.dict_start[operation][
            (thread_id, experiment_id)
        ] = self.current_milli_time()

    def end(self, operation, thread_id, experiment_id):
        self.dict_end[operation][(thread_id, experiment_id)] = self.current_milli_time()

    def print_times(self):
        res = ""
        average_dict = {}
        for op, dict_ in self.dict_start.items():
            n = 0
            for p, start_time in dict_.items():
                thread_id, experiment_id = p
                end_time = self.dict_end[op][p]
                total_time = end_time - start_time
                res += f"Thread: {thread_id} experiment: {experiment_id} operation: {op} time: {total_time}ms\n"
                if op in average_dict.keys():
                    average_dict[op] = (total_time + (n * average_dict[op])) / (n + 1)
                else:
                    average_dict[op] = total_time
                n += 1
        for op, average in average_dict.items():
            res += f"{op} average time: {average}ms\n"
        if not res == "":
            print(res[:-1])
