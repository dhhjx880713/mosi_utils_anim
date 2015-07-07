'''
Created on Mar 23, 2015
https://pythonhosted.org/watchdog/quickstart.html#quickstart

'''


import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import threading
class CustomLoggingEventHandler(PatternMatchingEventHandler):
    """Copied from LoggingEventHandler. Logs all the events captured for files
       matching patterns."""
    def __init__(self, patterns=None, ignore_patterns=None,
                 ignore_directories=False, case_sensitive=False,
                 callback=None):
        super(CustomLoggingEventHandler, self).__init__(patterns,
                                                          ignore_patterns,
                                                          ignore_directories,
                                                          case_sensitive)
        self.callback = callback
        
        
    def on_moved(self, event):
        super(CustomLoggingEventHandler, self).on_moved(event)

        what = 'directory' if event.is_directory else 'file'
        logging.info("Moved %s: from %s to %s", what, event.src_path,
                     event.dest_path)
        if self.callback != None:
            self.callback(event.dest_path)

    def on_created(self, event):
        super(CustomLoggingEventHandler, self).on_created(event)

        what = 'directory' if event.is_directory else 'file'
        logging.info("Created %s: %s", what, event.src_path)
        if self.callback != None:
            self.callback(event.src_path)

    def on_deleted(self, event):
        super(CustomLoggingEventHandler, self).on_deleted(event)

        what = 'directory' if event.is_directory else 'file'
        logging.info("Deleted %s: %s", what, event.src_path)

    def on_modified(self, event):
        super(CustomLoggingEventHandler, self).on_modified(event)

        what = 'directory' if event.is_directory else 'file'
        logging.info("Modified %s: %s", what, event.src_path)
        if self.callback != None:
            self.callback(event.src_path)

def start_directory_observation(path,event_handler):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
  
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def create_directory_observer(path,callback=None,recursive=False):
    event_handler = CustomLoggingEventHandler(patterns=["*.bvh"],callback=callback)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
  
    observer = Observer()
    observer.schedule(event_handler, path, recursive=recursive)
   
    return observer
    

class DirectoryObserverThread(threading.Thread):
    '''
    controls a CustomLoggingEventHandler
    '''
    def __init__(self, path,callback=None):
        threading.Thread.__init__(self)
        self.observer = create_directory_observer(path,callback=callback)

               
    def run(self):
        print "starting the observer"
        self.observer.start()

    def stop(self):
        print "stopping server"
        self.observer.stop()



if __name__ == "__main__":
    event_handler = CustomLoggingEventHandler(patterns=["*.bvh"])
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    start_directory_observation(path,event_handler)
    
    
    