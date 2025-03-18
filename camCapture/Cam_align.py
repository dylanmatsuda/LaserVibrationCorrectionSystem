def start_processing(self):
    """ Start the frame processing in a separate thread. """
    global processing
    if not processing:
        processing_thread = threading.Thread(target=process_frame, args=(self,), daemon=True)
        processing_thread.start()


def stop_processing(self):
    """ Stop the frame processing. """
    global processing
    processing = False
