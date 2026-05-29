import pickle

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

while True:
    message = input("\nEnter a message: ")

    prediction = model.predict([message])[0]

    if prediction == "ham":
        prediction = "not spam"

    print("Prediction:", prediction)