# Replicant 🤖✨  
Have you ever thought, *"What if I could train machine learning models to mimic my functions?"* If yes, then meet **Replicant**, the python-package that allows you to inject ML and AI into your code seamlessly. 🚀

## What Does It Do? 🤔

Replicant takes your perfectly functional, but extremely ordinary and boring, Python functions and replaces them with high-tech, VC-approved, machine learning models. Why? Because who needs simplicity when you can have a Decision Tree pretending to be your `if` statements? 🌳🤷‍♂️ All the billionaire tech-geniuses are telling you to embrace AI. Replicant enables you to do so with ️one line of code.

### Key Features 🌟

- **Function Mimicry**: Wrap your functions in machine learning models that do the same thing, but worse. 🦾
- **Synthetic Data Generation**: Why use real data when you can generate fake data to train your fake models? 🌀
- **Automatic Input Space Inference**: Replicant will analyze your function's logic and guess what kind of inputs it expects. It's like a psychic 🔮, but for Python 🐍.
- **Custom Models**: Bring your own absurdity by plugging in custom ML models. Want a neural network to decide if `x > 5`? Go for it. 🤖🧠

## Installation 🛠️
This package uses `uv` for dependency management, so make sure you have that installed. Then simply run:

```bash
make setup
```

(or `make setup-dev` if you're feeling bold 💪) followed by:

```bash
make build
```

🎉 Congratulations, your start-up just became a unicorn. 🦄

## Usage 📚
### Step 1: Write a Function ✍️
Write a perfectly good function that works just fine on its own.

### Step 2: Destroy Its Simplicity 💥
Use the @mimic decorator to replace your function with a machine learning model. Why? Because you can. 😎

```PYTHON
from replicant import mimic

@mimic(task="regressor", input_space={"x": (-10, 10)}, n_samples=10_00)
def square(x):
    return x ** 2
```

Now, instead of just squaring numbers, your function will train a model to square numbers. It’s like hiring a team of consultants to do basic arithmetic. 🤓

### Step 3: Enjoy the Chaos 🌀
Call your function and marvel at how much slower and more complicated it is now. 🤯

```PYTHON
result = square(4)
print(result)  # Should be close to 16, but who knows? 🤷‍♀️
```

## License 📜
This project is licensed under the MIT License, because even absurdity deserves freedom. 🕊️

---

Replicant: Because sometimes, doing things the hard way is the only way. 🛤️