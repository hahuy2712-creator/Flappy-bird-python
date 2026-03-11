import pygame
import random
import cv2
import sys
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize pygame
pygame.init()

# Game Constants
WIDTH, HEIGHT = 800, 600
BLOCK_SIZE = 20
FPS = 8

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 0, 0) # For debug deadzone

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Snake Game")
clock = pygame.time.Clock()

# MediaPipe Hand Tracking Setup using Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       running_mode=vision.RunningMode.VIDEO,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# OpenCV Camera Setup
cap = cv2.VideoCapture(0)

# Directions
UP = (0, -BLOCK_SIZE)
DOWN = (0, BLOCK_SIZE)
LEFT = (-BLOCK_SIZE, 0)
RIGHT = (BLOCK_SIZE, 0)

font = pygame.font.SysFont("comicsansms", 24)
large_font = pygame.font.SysFont("comicsansms", 50)

def spawn_food(snake_body, is_large=False):
    while True:
        max_x = (WIDTH - BLOCK_SIZE) // BLOCK_SIZE
        max_y = (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE
        if is_large:
            max_x -= 1 # Prevent spawning partially out of bounds
            max_y -= 1
        x = random.randint(0, max_x) * BLOCK_SIZE
        y = random.randint(0, max_y) * BLOCK_SIZE
        
        # Check collision with snake
        valid = True
        if is_large:
            blocks = [[x, y], [x + BLOCK_SIZE, y], [x, y + BLOCK_SIZE], [x + BLOCK_SIZE, y + BLOCK_SIZE]]
            for b in blocks:
                if b in snake_body:
                    valid = False
        else:
            if [x, y] in snake_body:
                valid = False
                
        if valid:
            return [x, y]

def main():
    snake = [
        [WIDTH // 2, HEIGHT // 2],
        [WIDTH // 2 - BLOCK_SIZE, HEIGHT // 2],
        [WIDTH // 2 - 2 * BLOCK_SIZE, HEIGHT // 2]
    ]
    direction = RIGHT
    score = 0
    small_foods_eaten = 0
    
    food = spawn_food(snake)
    large_food = None
    large_food_spawned = False
    
    running = True
    game_over = False
    last_ts = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != DOWN:
                    direction = UP
                if event.key == pygame.K_DOWN and direction != UP:
                    direction = DOWN
                if event.key == pygame.K_LEFT and direction != RIGHT:
                    direction = LEFT
                if event.key == pygame.K_RIGHT and direction != LEFT:
                    direction = RIGHT
                if event.key == pygame.K_r and game_over:
                    main() # restart
                    return
                if event.key == pygame.K_q and game_over:
                    running = False

        # Camera process
        ret, frame = cap.read()
        pip_surface = None
        if ret:
            frame = cv2.flip(frame, 1) # mirror
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            ts = int(time.time() * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect_for_video(mp_image, ts)
            
            threshold = 50 # deadzone threshold
            
            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                l8 = hand_landmarks[8] # INDEX_FINGER_TIP
                cx, cy = int(l8.x * w), int(l8.y * h)
                
                # Draw index finger tip manually
                cv2.circle(rgb_frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                
                dx = cx - w // 2
                dy = cy - h // 2
                
                if abs(dx) > threshold or abs(dy) > threshold:
                    if abs(dx) > abs(dy):
                        if dx > 0 and direction != LEFT:
                            direction = RIGHT
                        elif dx < 0 and direction != RIGHT:
                            direction = LEFT
                    else:
                        if dy > 0 and direction != UP:
                            direction = DOWN
                        elif dy < 0 and direction != DOWN:
                            direction = UP

            # Draw deadzone on PIP frame for user's reference
            cv2.rectangle(rgb_frame, (w//2 - threshold, h//2 - threshold), (w//2 + threshold, h//2 + threshold), RED, 2)
            cv2.circle(rgb_frame, (w//2, h//2), 3, RED, -1)

            # Convert to PIP surface
            pip_w = 200
            pip_h = int(200 * (h / w))
            pip_frame = cv2.resize(rgb_frame, (pip_w, pip_h))
            pip_frame = np.rot90(pip_frame)
            pip_frame = np.flipud(pip_frame)
            pip_surface = pygame.surfarray.make_surface(pip_frame)

        if not game_over:
            head = [snake[0][0] + direction[0], snake[0][1] + direction[1]]
            
            # Wall collision
            if head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT:
                game_over = True
                
            # Self collision
            if head in snake[1:]:
                game_over = True
                
            if not game_over:
                snake.insert(0, head)
                eaten = False
                
                # Check small food collision
                if head == food:
                    score += 1
                    small_foods_eaten += 1
                    food = spawn_food(snake)
                    eaten = True
                    
                    if small_foods_eaten > 0 and small_foods_eaten % 10 == 0 and not large_food_spawned:
                        large_food = spawn_food(snake, is_large=True)
                        large_food_spawned = True

                # Check large food collision
                elif large_food_spawned and large_food:
                    lx, ly = large_food
                    # Large food is a 2x2 grid
                    if head[0] in (lx, lx + BLOCK_SIZE) and head[1] in (ly, ly + BLOCK_SIZE):
                        score += 5
                        large_food = None
                        large_food_spawned = False
                        eaten = True

                if not eaten:
                    snake.pop()

        # Rendering
        screen.fill(BLACK)
        
        # Draw small food
        pygame.draw.rect(screen, WHITE, [food[0], food[1], BLOCK_SIZE, BLOCK_SIZE])
        
        # Draw large food
        if large_food_spawned and large_food:
            pygame.draw.rect(screen, WHITE, [large_food[0], large_food[1], BLOCK_SIZE * 2, BLOCK_SIZE * 2])
            
        # Draw snake
        for i, block in enumerate(snake):
            pygame.draw.rect(screen, WHITE, [block[0], block[1], BLOCK_SIZE, BLOCK_SIZE])
            pygame.draw.rect(screen, BLACK, [block[0]+1, block[1]+1, BLOCK_SIZE-2, BLOCK_SIZE-2], 1)

        # Draw UI
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        if pip_surface:
            pip_rect = pip_surface.get_rect()
            pip_rect.topright = (WIDTH - 10, 10)
            screen.blit(pip_surface, pip_rect)
            pygame.draw.rect(screen, WHITE, pip_rect, 2)
            
        if game_over:
            text = large_font.render("GAME OVER", True, WHITE)
            subtext = font.render("Press 'R' to Restart or 'Q' to Quit", True, WHITE)
            screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()))
            screen.blit(subtext, (WIDTH//2 - subtext.get_width()//2, HEIGHT//2))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
    cap.release()
    pygame.quit()
    sys.exit()
