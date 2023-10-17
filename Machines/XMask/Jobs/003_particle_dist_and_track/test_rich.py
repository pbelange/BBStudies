from rich.progress import Progress, BarColumn, TextColumn,TimeElapsedColumn,SpinnerColumn,TimeRemainingColumn
import time
import matplotlib.pyplot as plot


class test_spinner():

    def __init__(self,):
        # Progress info
        #-------------------------
        self.progress  = False
        self._plive    = None
        self._pstatus  = None
        #-------------------------

        try:
            self.main_test()
        except Exception as error:
            self.closeLiveDisplay()
            print("An error occurred:", type(error).__name__, "–", error)
        except KeyboardInterrupt:
            self.closeLiveDisplay()
            print("Terminated by user: KeyboardInterrupt")

    def main_test(self,):
        # self._plive.update(self._pstatus, advance=1,update=True)
        self.startSpinner()
        # self._plive.render(3)
        time.sleep(3)  
        # plt.pause(3)
        self.updateLive()
        self.closeLiveDisplay()


        # Progress bar methods
    #=============================================================================
    def startProgressBar(self,):
        self._plive = Progress("{task.description}",
                                TextColumn("[progress.remaining] ["),TimeRemainingColumn(),TextColumn("[progress.remaining]remaining ]   "),
                                SpinnerColumn(),
                                BarColumn(bar_width=40),
                                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                                TimeElapsedColumn())

        self._plive.start()
        self._plive.live._disable_redirect_io()

        self._pstatus = self._plive.add_task("[blue]Tracking\n", total=self.n_turns)
    
    def updateProgressBar(self,):
        self._plive.update(self._pstatus, advance=1,update=True)

    def updateLive(self,):
        self._plive.update(self._pstatus,advance=1,update=True)
        
    def startSpinner(self,):

        self._plive = Progress("{task.description}",
                                SpinnerColumn('aesthetic'),
                                TextColumn("[progress.elapsed] ["),TimeElapsedColumn (),TextColumn("[progress.elapsed]elapsed ]   "))

        self._plive.start()
        self._plive.live._disable_redirect_io()

        self._pstatus = self._plive.add_task("[blue]Tracking")


    def closeLiveDisplay(self,):
        self._plive.refresh()
        self._plive.stop()
        self._plive.console.clear_live()


# test_spinner()