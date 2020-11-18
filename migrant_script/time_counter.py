import time;

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
            "task.update_task": {}
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
            "task.update_task": {}

        }
        self.current_milli_time = lambda: int(round(time.time() * 1000))

    def start(self, operation ,thread_id, experiment_id):
        self.dict_start[operation][(thread_id,experiment_id)] = self.current_milli_time()
    def end(self, operation ,thread_id, experiment_id):
        self.dict_end[operation][(thread_id,experiment_id)] = self.current_milli_time()

    def print_times(self):
        res = ''
        average_dict = {}
        for op,dict in self.dict_start.items():
            n = 0
            for p,start_time in dict.items():
                thread_id,experiment_id = p
                end_time = self.dict_end[op][p]
                total_time = end_time - start_time
                res += 'Thread: ' +str(thread_id) + ' experiment: ' + str(experiment_id) + ' operation: '+ op + ' time: '+ str(total_time) +' millisecond.\n'
                if op in average_dict.keys():
                    average_dict[op] = (total_time + (n * average_dict[op]))/(n+1)
                else:
                    average_dict[op] = total_time
                n+=1
        for op,average in average_dict.items():
            res+= op +' average time: ' + str(average) + ' millisecond\n'
        if not res=='':
            print(res[:-1])