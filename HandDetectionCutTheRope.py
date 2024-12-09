import cv2
import mediapipe as mp
import pygame
import pymunk
import pymunk.pygame_util
from pygame.locals import QUIT
import math

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# Pymunk setup
space = pymunk.Space()
space.gravity = (0, 900)
draw_options = pymunk.pygame_util.DrawOptions(screen)

def create_rope(space, start_pos, length=10, segment_length=20):
    """Creates a rope with tightly connected segments, anchored at the top."""
    segments = []
    static_body = space.static_body  # Static body to anchor the rope
    prev_body = static_body

    for i in range(length):
        body = pymunk.Body()
        body.position = start_pos[0], start_pos[1] + i * segment_length
        shape = pymunk.Segment(body, (0, 0), (0, segment_length), 2)
        shape.density = 1
        shape.elasticity = 0.1  # Reduce elasticity for stability
        space.add(body, shape)
        segments.append(shape)
        
        # Connect the segment to the previous body with tight joints
        joint = pymunk.PivotJoint(prev_body, body, (start_pos[0], start_pos[1] + i * segment_length))
        space.add(joint)

        # Add a rotational limit to keep the rope segments stable
        rotational_limit = pymunk.RotaryLimitJoint(prev_body, body, -0.5, 0.5)  # Limits the rotation
        space.add(rotational_limit)
        
        prev_body = body

    return segments

# Create ropes
ropes = [
    create_rope(space, (200, 100)),
    create_rope(space, (400, 100)),
    create_rope(space, (600, 100)),
]

# Add candy object
candy_body = pymunk.Body()
candy_body.position = (400, 100 + len(ropes[1]) * 20)
candy_shape = pymunk.Circle(candy_body, 15)
candy_shape.density = 1
candy_shape.elasticity = 0.5
space.add(candy_body, candy_shape)

# Attach candy to the bottom of the middle rope
candy_joint = pymunk.PinJoint(candy_body, ropes[1][-1].body, (0, 0), (0, 25))
space.add(candy_joint)

# Goal area
goal_rect = pygame.Rect(300, 500, 200, 50)

# Game loop
running = True
win = False
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw goal area
    pygame.draw.rect(screen, (0, 255, 0), goal_rect)

    # Draw ropes
    space.debug_draw(draw_options)

    # Draw candy (with safety check)
    candy_x, candy_y = candy_body.position
    if not (math.isnan(candy_x) or math.isnan(candy_y)):
        pygame.draw.circle(screen, (255, 165, 0), (int(candy_x), int(candy_y)), 15)
    else:
        print("Candy position invalid! Resetting simulation...")
        running = False  # Exit the game or handle gracefully

    def is_cutting_motion(landmarks):
        """
        Determines if the index and middle fingers are extended and making a cutting motion.
        """
        # Index and middle finger tip positions
        index_tip = landmarks[8]
        middle_tip = landmarks[12]

        # Positions of the DIP (Distal Interphalangeal) joints for comparison
        index_dip = landmarks[7]
        middle_dip = landmarks[11]

        # Check if fingers are extended: TIP is above DIP in the y-axis (screen coordinate system is inverted)
        index_extended = index_tip.y < index_dip.y
        middle_extended = middle_tip.y < middle_dip.y

        # Define additional cutting logic (e.g., vertical or horizontal motion)
        # Example: Check if both fingers move downward significantly
        motion = index_tip.y - middle_tip.y
        cutting_motion = index_extended and middle_extended and abs(motion) < 0.02  # Adjust the threshold

        return cutting_motion


    # Main game loop (update hand detection logic)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                pygame.draw.circle(screen, (0, 255, 0), (x, y), 5)
                
            if is_cutting_motion(landmarks):
                # Calculate position of the index finger (for cutting collision detection)
                index_x = int(landmarks[8].x * WIDTH)
                index_y = int(landmarks[8].y * HEIGHT)

                pygame.draw.circle(screen, (255, 0, 0), (index_x, index_y), 10)  # Visualize cutting point
                

                # Check if the cutting point intersects with any rope segment
                for rope in ropes:
                    for segment in rope:
                        p1 = segment.body.position + segment.a
                        p2 = segment.body.position + segment.b
                        if pygame.math.Vector2(index_x, index_y).distance_to((p1.x, p1.y)) < 10 or \
                           pygame.math.Vector2(index_x, index_y).distance_to((p2.x, p2.y)) < 10:
                            # Remove the segment and its joints safely
                            for joint in space.constraints:
                                if joint.a == segment.body or joint.b == segment.body:
                                    space.remove(joint)
                            space.remove(segment.body, segment)
                            rope.remove(segment)


    # Check for win condition
    if goal_rect.collidepoint(candy_body.position.x, candy_body.position.y):
        win = True

    # Display win message
    if win:
        text = font.render("You Win!", True, (255, 255, 255))
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
    else:
        # Step physics
        space.step(1 / 60.0)

    pygame.display.flip()
    clock.tick(60)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
