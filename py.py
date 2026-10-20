import turtle
import random

# Screen setup
screen = turtle.Screen()
screen.setup(width=800, height=600)
screen.bgcolor("black")
screen.title("Random Walk Landscape")

# Turtle setup
walker = turtle.Turtle()
walker.speed(0)  # Fastest animation speed
walker.width(3)  # Line thickness
walker.hideturtle()

# List of vibrant landscape colors
colors = ["#39FF14", "#0FF0FC", "#FF007F", "#BC13FE", "#FFEA00", "#FF5F1F"]

# Constraints for the walk
step_length = 20
angles = [0, 45, 90, 135, 180, 225, 270, 315]

# Main loop to generate the landscape
while True:
    # 1. Pick a random color and direction
    walker.color(random.choice(colors))
    walker.setheading(random.choice(angles))
    
    # 2. Take a step forward
    walker.forward(step_length)
    
    # 3. Teleport back to center if the walker wanders too far off-screen
    if abs(walker.xcor()) > 380 or abs(walker.ycor()) > 280:
        walker.penup()
        walker.goto(0, 0)
        walker.pendown()
