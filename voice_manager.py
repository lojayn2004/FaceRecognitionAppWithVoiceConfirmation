import pygame
import threading
import time
import os

class VoiceManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {}
        self.last_played = 0
        self.cooldown = 3  
        self.sound_files = {
            
            "marked": "sounds/marked.mp3", 
            "unknown": "sounds/unknown.mp3"
        }
        
        self._load_sounds()
    
    def _load_sounds(self):
        for name, filepath in self.sound_files.items():
            if os.path.exists(filepath):
                try:
                    self.sounds[name] = pygame.mixer.Sound(filepath)
                    print(f"Loaded: {filepath}")
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")
            else:
                print(f"File not found: {filepath}")
    
    def play(self, sound_type):
       
        def _play():
            try:
                
                current_time = time.time()
                if current_time - self.last_played < self.cooldown:
                    return
                
                if sound_type in self.sounds:
                    self.sounds[sound_type].play()
                    self.last_played = current_time
                    print(f"Playing: {sound_type}")
                else:
                    print(f"Sound not loaded: {sound_type}")
                    
            except Exception as e:
                print(f"Audio error: {e}")
        
      
        thread = threading.Thread(target=_play)
        thread.daemon = True
        thread.start()

