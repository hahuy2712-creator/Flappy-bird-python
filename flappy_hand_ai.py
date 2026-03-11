import pygame
import random
import cv2
import mediapipe as mp

WIDTH = 400
HEIGHT = 600
FPS = 60

GRAVITY = 0.5
JUMP = -9

PIPE_WIDTH = 70
PIPE_GAP = 170
PIPE_SPEED = 3

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird Gesture PRO")

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial",30)
big_font = pygame.font.SysFont("Arial",50)

# COLORS
SKY = (135,206,235)
GREEN = (0,200,0)
YELLOW = (255,220,0)
BLACK = (0,0,0)

# HAND TRACKING
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# GAME STATE
bird_y = HEIGHT//2
bird_vel = 0

pipe_x = WIDTH
pipe_height = random.randint(150,400)

score = 0
high_score = 0

game_over = False


def count_fingers(hand_landmarks):

    tips = [4,8,12,16,20]
    fingers = 0

    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
            fingers += 1

    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers += 1

    return fingers


def detect_gesture():

    ret, frame = cap.read()
    if not ret:
        return 0

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    fingers = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(hand_landmarks)

    cv2.imshow("Camera Control", frame)
    cv2.waitKey(1)

    return fingers


def reset_game():

    global bird_y, bird_vel, pipe_x, pipe_height, score, game_over

    bird_y = HEIGHT//2
    bird_vel = 0
    pipe_x = WIDTH
    pipe_height = random.randint(150,400)

    score = 0
    game_over = False


running = True

while running:

    clock.tick(FPS)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and game_over:
                reset_game()

    if not game_over:

        fingers = detect_gesture()

        # GESTURE CONTROL
        if fingers >= 4:      # open hand
            bird_vel = JUMP

        if fingers == 0:      # fist
            bird_vel += GRAVITY*2

        bird_vel += GRAVITY
        bird_y += bird_vel

        pipe_x -= PIPE_SPEED

        if pipe_x < -PIPE_WIDTH:
            pipe_x = WIDTH
            pipe_height = random.randint(150,400)
            score += 1

            if score > high_score:
                high_score = score

        # COLLISION
        if bird_y < 0 or bird_y > HEIGHT:
            game_over = True

        if 80+18 > pipe_x and 80-18 < pipe_x+PIPE_WIDTH:
            if bird_y-18 < pipe_height or bird_y+18 > pipe_height+PIPE_GAP:
                game_over = True

    # DRAW
    screen.fill(SKY)

    pygame.draw.circle(screen,YELLOW,(80,int(bird_y)),18)

    pygame.draw.rect(screen,GREEN,(pipe_x,0,PIPE_WIDTH,pipe_height))
    pygame.draw.rect(screen,GREEN,(pipe_x,pipe_height+PIPE_GAP,PIPE_WIDTH,HEIGHT))

    score_text = font.render("Score: "+str(score),True,BLACK)
    high_text = font.render("High: "+str(high_score),True,BLACK)

    screen.blit(score_text,(10,10))
    screen.blit(high_text,(10,40))

    if game_over:

        over_text = big_font.render("GAME OVER",True,(200,0,0))
        restart_text = font.render("Press R to Restart",True,BLACK)

        screen.blit(over_text,(70,250))
        screen.blit(restart_text,(100,320))

    pygame.display.update()

pygame.quit()
cap.release()
cv2.destroyAllWindows()