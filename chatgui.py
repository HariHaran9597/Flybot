from tkinter import Tk, Frame, Scrollbar, Label, END, Entry, Text, VERTICAL, Button, RIGHT, Y
from flybot import Chatbot
import json

class ChatGUI:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.window = Tk()
        self.window.title("Flybot")
        self.window.geometry("400x500")
        self.window.resizable(width=False, height=False)

        self.chat_frame = Frame(self.window)
        self.chat_frame.pack(expand=True, fill='both')

        self.text_widget = Text(self.chat_frame, wrap="word")
        self.text_widget.pack(expand=True, fill='both')

        self.scrollbar = Scrollbar(self.chat_frame, orient=VERTICAL, command=self.text_widget.yview)
        self.scrollbar.pack(side=RIGHT, fill=Y)

        self.text_widget.config(yscrollcommand=self.scrollbar.set)
        self.text_widget.bind('<KeyPress>', lambda e: 'break')

        self.input_frame = Frame(self.window)
        self.input_frame.pack(expand=True, fill='both')

        self.input_field = Entry(self.input_frame)
        self.input_field.pack(side="left", expand=True, fill='both')

        self.send_button = Button(self.input_frame, text="Send", command=self.send, bg="white", fg="black")
        self.send_button.pack(side="left", padx=5)

        self.input_field.bind("<Return>", self.send)

    def send(self, event=None):
        user_input = self.input_field.get()
        self.input_field.delete(0, END)
        if user_input:
            response = self.chatbot.get_response(user_input)
            self.text_widget.insert(END, "User: " + user_input + "\n\n")
            self.text_widget.insert(END, "Flybot: " + response + "\n\n")
            self.text_widget.see(END)

if __name__ == "__main__":
    with open("intents.json", "r") as file:
        dataset = json.load(file)

    chatbot = Chatbot(dataset)
    chatbot.preprocess_data()

    gui = ChatGUI(chatbot)
    gui.window.mainloop()
