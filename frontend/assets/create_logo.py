# create_logo.py — run from the repo or frontend dir
import base64, os
b64 = b"""iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAABmJLR0QA/wD/AP+gvaeTAAABT0lEQVR4nO3YwQnCQBRE0c5n4/0m0XQkQmKq2kY6CP4RzQz5jQ8j5g1gqFj3z4f3z8b5ZlM8s7Z7u7s6ZZmZmbm5u3Xq9Xq9Xq9Xo9Xo9Xq9Xq9Xp9b3p+TgC2gE1gC2gE1gC2gE1gC2gE1gC2gE1gC2gE1gC2gE1gC2gE1gC2gE1gK6wJzjvA9oAtoBNYAtqATWALagE1gC2oBNYAtqATWALagE1gC2oBNYAtqATWALagE1gC2oBNYAtqATWALagE1gC2oBNYAtqATWALagE1gK9iQ9p8u0Ew4AAAAASUVORK5CYII="""
out = "frontend/assets/logo.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as f:
    f.write(base64.b64decode(b64))
print("Wrote", out)
