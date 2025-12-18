#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Battle Visualizer - Read serial data and display battle results
Visualize inference results from devices A and B using Pygame
"""

import re
import sys
from datetime import datetime
import os

# Import serial (pyserial)
try:
    import serial
    # Check if Serial class exists (pyserial)
    if not hasattr(serial, 'Serial'):
        print("Error: Wrong 'serial' module imported!")
        print("Please install pyserial with: pip install pyserial")
        print("Note: There may be a conflicting 'serial' package installed.")
        sys.exit(1)
except ImportError:
    print("Error: pyserial not installed!")
    print("Please install it with: pip install pyserial")
    sys.exit(1)

# Import pygame
try:
    import pygame
except ImportError:
    print("Error: pygame not installed!")
    print("Please install it with: pip install pygame")
    sys.exit(1)

# ========== Configuration ==========
# Serial port configuration
SERIAL_PORT = '/dev/tty.usbmodem12101'  # Specific port, or use None for auto-detect
BAUD_RATE = 9600
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
BLUE = (50, 150, 255)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 50)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)

# Gesture names and color mapping
GESTURES = {
    0: ("Circle (Wood)", (139, 69, 19)),      # Brown
    1: ("Other (None)", GRAY),
    2: ("Peak (Fire)", RED),                  # Red
    3: ("Wave (Water)", BLUE)                 # Blue
}

# Battle result colors
RESULT_COLORS = {
    "A Wins": RED,
    "B Wins": BLUE,
    "Draw": YELLOW,
    "Invalid Match": GRAY
}

class BattleVisualizer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("IMU Battle Visualizer - Serial Monitor")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        self.font_tiny = pygame.font.Font(None, 24)
        
        # 数据存储
        self.resultA = None
        self.resultB = None
        self.battleResult = None
        self.timeDiff = None
        self.lastUpdate = None
        self.lastBattleTime = None  # Time when last battle result was shown
        self.matchCount = 0
        self.history = []  # Store history records
        self.displayDuration = 10.0  # Display battle result for 10 seconds
        
        # Serial port connection
        self.ser = None
        self.connect_serial()
        
    def connect_serial(self):
        """Connect to serial port"""
        import glob
        import time
        
        # Don't reconnect too frequently
        if hasattr(self, '_last_reconnect_time'):
            if time.time() - self._last_reconnect_time < 2:
                return False
        self._last_reconnect_time = time.time()
        
        try:
            # If specific port is configured, try it first
            if SERIAL_PORT and '*' not in SERIAL_PORT:
                try:
                    if self.ser and self.ser.is_open:
                        self.ser.close()
                    self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                    print(f"Successfully connected to serial port: {SERIAL_PORT}")
                    return True
                except (OSError, serial.SerialException) as e:
                    # Silently fail if device not available
                    if getattr(e, 'errno', None) != 6:
                        print(f"Failed to connect to {SERIAL_PORT}: {e}")
                    return False
                except Exception as e:
                    return False
            
            # Auto-detect serial ports
            if sys.platform.startswith('win'):
                ports = ['COM%s' % (i + 1) for i in range(256)]
            elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
                ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
            elif sys.platform.startswith('darwin'):
                ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/tty.usbserial*')
            else:
                raise EnvironmentError('Unsupported platform')
            
            if not ports:
                return False
                
            # Try to connect to available ports
            for port in ports:
                try:
                    if self.ser and self.ser.is_open:
                        self.ser.close()
                    self.ser = serial.Serial(port, BAUD_RATE, timeout=1)
                    print(f"Successfully connected to serial port: {port}")
                    return True
                except:
                    continue
                    
            return False
        except Exception as e:
            return False
    
    def parse_serial_line(self, line):
        """Parse serial data from Node B (Central)"""
        line = line.strip()
        
        if not line:
            return None
        
        # Parse battle result section header first (most specific)
        # Format: "--- BATTLE RESULT (Valid Match) ---"
        if "BATTLE RESULT (Valid Match)" in line:
            print(f"[DEBUG] Parsed battle_start from: {line}")
            return {"type": "battle_start"}
        
        # Parse NODE_A result in battle section
        # Format: "NODE_A: peak (100.00%) - Time: 236948 ms"
        match = re.search(r'NODE_A:\s*(\w+)\s*\(([\d.]+)%\)\s*-\s*Time:\s*(\d+)\s*ms', line)
        if match:
            gesture = match.group(1)
            confidence = float(match.group(2))
            timestamp = int(match.group(3))
            gesture_map = {"circle": 0, "other": 1, "peak": 2, "wave": 3}
            class_id = gesture_map.get(gesture, -1)
            print(f"[DEBUG] Parsed battle_resultA from: {line} -> class_id={class_id}, gesture={gesture}, confidence={confidence}")
            return {"type": "battle_resultA", "class_id": class_id, "gesture": gesture, 
                   "confidence": confidence, "timestamp": timestamp}
        
        # Parse NODE_B result in battle section
        # Format: "NODE_B: wave (99.65%) - Time: 237348 ms"
        match = re.search(r'NODE_B:\s*(\w+)\s*\(([\d.]+)%\)\s*-\s*Time:\s*(\d+)\s*ms', line)
        if match:
            gesture = match.group(1)
            confidence = float(match.group(2))
            timestamp = int(match.group(3))
            gesture_map = {"circle": 0, "other": 1, "peak": 2, "wave": 3}
            class_id = gesture_map.get(gesture, -1)
            print(f"[DEBUG] Parsed battle_resultB from: {line} -> class_id={class_id}, gesture={gesture}, confidence={confidence}")
            return {"type": "battle_resultB", "class_id": class_id, "gesture": gesture, 
                   "confidence": confidence, "timestamp": timestamp}
        
        # Parse time difference
        # Format: "Time Difference: 400 ms"
        match = re.search(r'Time Difference:\s*(\d+)\s*ms', line)
        if match:
            time_diff = int(match.group(1))
            print(f"[DEBUG] Parsed time_diff from: {line} -> {time_diff}")
            return {"type": "time_diff", "time_diff": time_diff}
        
        # Parse battle result
        # Format: "Result: B Wins"
        match = re.search(r'Result:\s*(.+)', line)
        if match:
            result = match.group(1).strip()
            print(f"[DEBUG] Parsed battle_result from: {line} -> {result}")
            return {"type": "battle_result", "result": result}
        
        # Parse inference result header from Node A
        # Format: "--- Inference Result from NODE_A ---"
        if "Inference Result from NODE_A" in line:
            return {"type": "result_start", "device": "A"}
        
        # Parse inference result header from Node B
        # Format: "--- Inference Result (Node B) ---"
        if "Inference Result (Node B)" in line:
            return {"type": "result_start", "device": "B"}
        
        # Parse predicted class
        # Format: "  Predicted Class: 0 (circle)"
        match = re.search(r'Predicted Class:\s*(\d+)\s*\((\w+)\)', line)
        if match:
            class_id = int(match.group(1))
            gesture = match.group(2)
            return {"type": "class", "class_id": class_id, "gesture": gesture}
        
        # Parse confidence
        # Format: " - Confidence: 85.23%"
        match = re.search(r'Confidence:\s*([\d.]+)%', line)
        if match:
            confidence = float(match.group(1))
            return {"type": "confidence", "confidence": confidence}
        
        # Parse timestamp
        # Format: " - Timestamp: 12345 ms" or " - Time: 12345 ms"
        match = re.search(r'(?:Timestamp|Time):\s*(\d+)\s*ms', line)
        if match:
            timestamp = int(match.group(1))
            return {"type": "timestamp", "timestamp": timestamp}
        
        # Parse invalid match
        if "Invalid Match" in line or "无效对决" in line:
            return {"type": "invalid_match"}
        
        return None
    
    def update_data(self, parsed_data, current_device):
        """Update data - only update display when battle result is complete"""
        if parsed_data is None:
            return
        
        # Only process battle result section, ignore individual inference results
        if parsed_data["type"] == "battle_start":
            # Start of battle result section - prepare for new battle data
            if not hasattr(self, '_tempResultA'):
                self._tempResultA = {}
            if not hasattr(self, '_tempResultB'):
                self._tempResultB = {}
            self._tempResultA = {}
            self._tempResultB = {}
            self._tempTimeDiff = None
            self._tempBattleResult = None
        
        elif parsed_data["type"] == "battle_resultA":
            # Update Node A result from battle section
            if not hasattr(self, '_tempResultA'):
                self._tempResultA = {}
            self._tempResultA = {
                "class_id": parsed_data["class_id"],
                "gesture": parsed_data["gesture"],
                "confidence": parsed_data["confidence"],
                "timestamp": parsed_data["timestamp"]
            }
        
        elif parsed_data["type"] == "battle_resultB":
            # Update Node B result from battle section
            if not hasattr(self, '_tempResultB'):
                self._tempResultB = {}
            self._tempResultB = {
                "class_id": parsed_data["class_id"],
                "gesture": parsed_data["gesture"],
                "confidence": parsed_data["confidence"],
                "timestamp": parsed_data["timestamp"]
            }
        
        elif parsed_data["type"] == "time_diff":
            if not hasattr(self, '_tempTimeDiff'):
                self._tempTimeDiff = None
            self._tempTimeDiff = parsed_data["time_diff"]
        
        elif parsed_data["type"] == "battle_result":
            # Only update display when we have the final battle result
            self._tempBattleResult = parsed_data["result"]
            
            # Update display only if we have complete battle data
            if hasattr(self, '_tempResultA') and hasattr(self, '_tempResultB') and \
               self._tempResultA and self._tempResultB:
                # Update display with new battle result
                self.resultA = self._tempResultA.copy()
                self.resultB = self._tempResultB.copy()
                self.timeDiff = self._tempTimeDiff
                self.battleResult = self._tempBattleResult
                self.matchCount += 1
                self.lastBattleTime = datetime.now()
                
                # Save to history
                self.history.append({
                    "resultA": self.resultA.copy(),
                    "resultB": self.resultB.copy(),
                    "battleResult": self.battleResult,
                    "timeDiff": self.timeDiff,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                # Keep only the last 10 records
                if len(self.history) > 10:
                    self.history.pop(0)
        
        elif parsed_data["type"] == "invalid_match":
            # Don't update display for invalid matches
            pass
    
    def draw_gesture_card(self, x, y, width, height, device_name, result, is_left=True):
        """Draw gesture card"""
        if result is None:
            # Draw waiting state
            pygame.draw.rect(self.screen, DARK_GRAY, (x, y, width, height))
            pygame.draw.rect(self.screen, GRAY, (x, y, width, height), 3)
            text = self.font_medium.render(f"{device_name}: Waiting...", True, WHITE)
            text_rect = text.get_rect(center=(x + width//2, y + height//2))
            self.screen.blit(text, text_rect)
            return
        
        # Get gesture information
        class_id = result.get("class_id", -1)
        confidence = result.get("confidence", 0.0)
        gesture_name, gesture_color = GESTURES.get(class_id, ("Unknown", GRAY))
        
        # Draw card background
        pygame.draw.rect(self.screen, BLACK, (x, y, width, height))
        pygame.draw.rect(self.screen, gesture_color, (x, y, width, height), 5)
        
        # Device name
        device_color = RED if is_left else BLUE
        title = self.font_medium.render(device_name, True, device_color)
        title_rect = title.get_rect(center=(x + width//2, y + 40))
        self.screen.blit(title, title_rect)
        
        # Gesture name
        gesture_text = self.font_large.render(gesture_name, True, gesture_color)
        gesture_rect = gesture_text.get_rect(center=(x + width//2, y + height//2 - 40))
        self.screen.blit(gesture_text, gesture_rect)
        
        # Confidence
        conf_text = self.font_medium.render(f"{confidence:.1f}%", True, WHITE)
        conf_rect = conf_text.get_rect(center=(x + width//2, y + height//2 + 40))
        self.screen.blit(conf_text, conf_rect)
        
        # Timestamp (if available)
        if "timestamp" in result:
            time_text = self.font_tiny.render(f"Time: {result['timestamp']} ms", True, GRAY)
            time_rect = time_text.get_rect(center=(x + width//2, y + height - 30))
            self.screen.blit(time_text, time_rect)
    
    def draw_battle_result(self, x, y, width, height):
        """Draw battle result"""
        if self.battleResult is None:
            return
        
        # Background
        result_color = RESULT_COLORS.get(self.battleResult, WHITE)
        pygame.draw.rect(self.screen, BLACK, (x, y, width, height))
        pygame.draw.rect(self.screen, result_color, (x, y, width, height), 5)
        
        # Result text
        result_text = self.font_large.render(self.battleResult, True, result_color)
        result_rect = result_text.get_rect(center=(x + width//2, y + height//2 - 30))
        self.screen.blit(result_text, result_rect)
        
        # Time difference
        if self.timeDiff is not None:
            time_text = self.font_small.render(f"Time Diff: {self.timeDiff} ms", True, WHITE)
            time_rect = time_text.get_rect(center=(x + width//2, y + height//2 + 30))
            self.screen.blit(time_text, time_rect)
    
    def draw_history(self, x, y, width, height):
        """Draw history records"""
        pygame.draw.rect(self.screen, DARK_GRAY, (x, y, width, height))
        pygame.draw.rect(self.screen, GRAY, (x, y, width, height), 2)
        
        # Title
        title = self.font_small.render("History", True, WHITE)
        self.screen.blit(title, (x + 10, y + 10))
        
        # Display last 5 records
        start_idx = max(0, len(self.history) - 5)
        y_offset = y + 50
        for i in range(start_idx, len(self.history)):
            record = self.history[i]
            if i == len(self.history) - 1:
                # Highlight latest record
                pygame.draw.rect(self.screen, (50, 50, 50), (x + 5, y_offset - 5, width - 10, 60))
            
            rA = record["resultA"]
            rB = record["resultB"]
            gestureA, colorA = GESTURES.get(rA.get("class_id", -1), ("?", GRAY))
            gestureB, colorB = GESTURES.get(rB.get("class_id", -1), ("?", GRAY))
            
            text = f"{record['timestamp']} | A:{gestureA[:4]} vs B:{gestureB[:4]} | {record['battleResult']}"
            history_text = self.font_tiny.render(text, True, WHITE)
            self.screen.blit(history_text, (x + 10, y_offset))
            y_offset += 50
    
    def draw(self):
        """Draw interface"""
        self.screen.fill((20, 20, 30))  # Dark background
        
        # Title
        title = self.font_large.render("IMU Battle Visualizer", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH//2, 40))
        self.screen.blit(title, title_rect)
        
        # Match count
        count_text = self.font_small.render(f"Matches: {self.matchCount}", True, GRAY)
        self.screen.blit(count_text, (WINDOW_WIDTH - 250, 20))
        
        # Device A card (left)
        card_width = 400
        card_height = 300
        margin = 50
        self.draw_gesture_card(
            margin, 
            120, 
            card_width, 
            card_height, 
            "NODE_A", 
            self.resultA, 
            True
        )
        
        # Device B card (right)
        self.draw_gesture_card(
            WINDOW_WIDTH - margin - card_width, 
            120, 
            card_width, 
            card_height, 
            "NODE_B", 
            self.resultB, 
            False
        )
        
        # VS label
        vs_text = self.font_large.render("VS", True, YELLOW)
        vs_rect = vs_text.get_rect(center=(WINDOW_WIDTH//2, 270))
        self.screen.blit(vs_text, vs_rect)
        
        # Battle result (center)
        result_width = 400
        result_height = 150
        self.draw_battle_result(
            WINDOW_WIDTH//2 - result_width//2,
            450,
            result_width,
            result_height
        )
        
        # History (bottom)
        self.draw_history(
            WINDOW_WIDTH//2 - 300,
            620,
            600,
            160
        )
        
        # Connection status
        status_text = "Connected" if self.ser and self.ser.is_open else "Disconnected"
        status_color = GREEN if self.ser and self.ser.is_open else RED
        status = self.font_tiny.render(f"Serial: {status_text}", True, status_color)
        self.screen.blit(status, (10, WINDOW_HEIGHT - 30))
        
        pygame.display.flip()
    
    def run(self):
        """Main loop"""
        current_device = None
        running = True
        import time
        
        print("Visualizer started...")
        print("Press ESC or close window to exit")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Check if battle result should be cleared (after display duration)
            if self.lastBattleTime:
                elapsed = (datetime.now() - self.lastBattleTime).total_seconds()
                if elapsed > self.displayDuration:
                    # Keep results but clear battle result text after duration
                    # Don't clear resultA and resultB - keep them visible
                    pass
            
            # Read serial data
            if self.ser and self.ser.is_open:
                try:
                    if self.ser.in_waiting > 0:
                        line = self.ser.readline().decode('utf-8', errors='ignore')
                        # Print to terminal
                        if line.strip():
                            print(line.rstrip())
                        parsed = self.parse_serial_line(line)
                        if parsed:
                            self.update_data(parsed, current_device)
                            if parsed.get("type") == "result_start":
                                current_device = parsed.get("device")
                except (OSError, serial.SerialException) as e:
                    # Device disconnected or not configured
                    error_code = getattr(e, 'errno', None)
                    if error_code == 6:  # Device not configured
                        # Close and try to reconnect
                        try:
                            self.ser.close()
                        except:
                            pass
                        self.ser = None
                        # Try to reconnect after a delay
                        time.sleep(1)
                        self.connect_serial()
                    else:
                        # Other serial errors - print only occasionally
                        if not hasattr(self, '_last_error_time') or time.time() - self._last_error_time > 5:
                            print(f"Serial read error: {e}")
                            self._last_error_time = time.time()
                except Exception as e:
                    # Other errors - print only occasionally
                    if not hasattr(self, '_last_error_time') or time.time() - self._last_error_time > 5:
                        print(f"Serial read error: {e}")
                        self._last_error_time = time.time()
            else:
                # Try to reconnect
                self.connect_serial()
            
            # Draw interface
            self.draw()
            self.clock.tick(30)  # 30 FPS
        
        # Cleanup
        if self.ser and self.ser.is_open:
            self.ser.close()
        pygame.quit()
        print("Program exited")

if __name__ == "__main__":
    visualizer = BattleVisualizer()
    visualizer.run()

