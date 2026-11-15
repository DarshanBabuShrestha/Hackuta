"""Local text-to-speech output for wearer-facing incident messages."""
import logging
import threading


_speech_lock = threading.Lock()
_engine_lock = threading.Lock()
_active_engine = None
_stop_requested = threading.Event()


def speak(message):
    """Speak a message synchronously so listening starts after it finishes."""
    try:
        import pyttsx3

        global _active_engine
        with _speech_lock:
            if _stop_requested.is_set():
                _stop_requested.clear()
                return False
            engine = pyttsx3.init()
            with _engine_lock:
                _active_engine = engine
            try:
                engine.say(message)
                engine.runAndWait()
            finally:
                try:
                    engine.stop()
                finally:
                    with _engine_lock:
                        if _active_engine is engine:
                            _active_engine = None
            interrupted = _stop_requested.is_set()
            _stop_requested.clear()
        return not interrupted
    except Exception as exc:
        _stop_requested.clear()
        logging.warning('System voice unavailable: %s', exc)
        return False


def stop_speaking():
    """Stop the message currently playing through the local speech engine."""
    _stop_requested.set()
    with _engine_lock:
        engine = _active_engine
    if engine is None:
        return True
    try:
        engine.stop()
        return True
    except Exception as exc:
        logging.warning('Could not stop system voice: %s', exc)
        return False
