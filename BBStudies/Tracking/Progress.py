
import time
import rich.progress as rp
import rich.live as rl
import rich.panel as rpn
import rich.table as rt
import traceback
# Progress, BarColumn, TextColumn,TimeElapsedColumn,SpinnerColumn,TimeRemainingColumn
from time import sleep



class ProgressBar():
    def __init__(self,Progress = None,message = 'Tracking',color='blue',n_steps = 1,mode='fixed',max_visible = 3):

        if Progress is None:
            self.Progress = rp.Progress("{task.description}",
                                        rp.TextColumn("[progress.remaining] ["),
                                        rp.TimeRemainingColumn(),rp.TextColumn("[progress.remaining]remaining ]   "),
                                        rp.SpinnerColumn(),
                                        rp.BarColumn(bar_width=40),
                                        rp.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                                        rp.TimeElapsedColumn())
        
        

        
        self.mode = mode
        self.message = message
        self.color   = color
        self.n_steps = n_steps
        self.Progress.add_task(f"[{self.color}]{self.message}\n", total=self.n_steps,start=False)
        self.main_task = self.Progress.tasks[-1]
        self.max_visible = max_visible


        
        self.console = self.Progress.console
        self.subtasks  = []
        # self.subtasks = []

    def add_subtask(self,subtask_name,message = 'subtask',color = 'red',n_steps = 0,level=1,start=True):
        
        self.Progress.add_task(f"[{color}]" + level*4*' ' + f"{message}\n", total=n_steps,start=start)
        
        self.subtasks.append(self.Progress.tasks[-1]) 
        if self.mode == 'append':
            self.n_steps = len(self.subtasks)

        # return subtask

    
    
    def start(self,):
        self.Progress.start_task(self.main_task.id)
        self.Progress.start()
        self.Progress.live._disable_redirect_io()
        

    def start_task(self,task_id):
        self.Progress.start_task(self.subtasks[task_id].id)
        # self.Progress.live._disable_redirect_io()


    def update_visibility(self,):
        for task in self.subtasks[:-self.max_visible]:
            self.Progress.update(task.id,visible=False)
            # if task.finished:
                # self.Progress.remove_task(task.id)
            # else:
                # task.visible = False

    def update(self,chunk = 1):
        if len(self.subtasks) > 0:
            self.Progress.update(self.subtasks[-1].id, advance=chunk,update=True,refresh=True)
        
            completed = sum([1 for subtask in self.subtasks if subtask.finished])
            # if self.subtasks[-1].finished:
            self.Progress.update(self.main_task.id, total = self.n_steps,completed=completed,update=True)
        else:
            self.Progress.update(self.main_task.id, advance=chunk,update=True)

        if self.max_visible is not None:
            self.update_visibility()

    def close(self,):
        # self.update()
        self.Progress.refresh()
        self.Progress.stop()
        self.Progress.console.clear_live()

        # # Saving execution time in seconds
        # self.exec_time = self.Progress.tasks[0].finished_time



# PROGRESS PANEL TO BE IMPLEMENTED

# class ProgressPanel():
#     def __init__(self,Progress = None,message = 'Tracking',color='blue',n_steps = 1,mode='append'):




#         self.progress_table = rt.Table.grid()
#         self.progress_table.add_row(rpn.Panel.fit(overall_progress  , title="Overall Progress", border_style="green", padding=(2, 2)),
#                                     rpn.Panel.fit(job_progress      , title="[b]Jobs", border_style="red", padding=(1, 2)))






if __name__ == "__main__":
    # testing
    # PBar = 

    iterable = range(10)

    PBar = ProgressBar(message='Test __________',color='blue',n_steps=10,mode='fixed')
    # PBar.add_subtask('job1',message = 'subtask1',color = 'red',n_steps = len(iterable),level=1,start=False)
    # PBar.add_subtask('job1',message = 'subtask1',color = 'red',n_steps = len(iterable),level=1,start=False)
    
    # PBar.add_subtask('job2',message = 'subtask2',color = 'red',n_steps = len(iterable),level=1)

    try:

        PBar.start()

        # Job 1
        for ii in range(10):

            PBar.add_subtask(f'job{ii}',message = f'subtask{ii}',color = 'red',n_steps = len(iterable),level=2)
            if not PBar.main_task.started:
                PBar.start()
            for i in iterable:
                time.sleep(0.1)
                PBar.update()

        # PBar.console.print('FIRST')
        # PBar.add_subtask('job2',message = 'subtask2',color = 'red',n_steps = len(iterable),level=1)
        # # PBar.start_task(1)
        # for i in iterable:
        #     time.sleep(0.1)
        #     PBar.update()
        
            if PBar.main_task.finished:
                PBar.close()
    
    except Exception as error:
        PBar.close()
        print("An error occurred:", type(error).__name__, " - ", error)
        traceback.print_exc()
    except KeyboardInterrupt:
        PBar.close()
        print("Terminated by user: KeyboardInterrupt")
