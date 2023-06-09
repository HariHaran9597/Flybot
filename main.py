from chatgui import Chatbot

def main():
    chatbot = Chatbot(data)
    chatbot.preprocess_data()
    chatbot.chat()

if __name__ == '__main__':
    main()
