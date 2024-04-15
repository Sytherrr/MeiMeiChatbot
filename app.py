from flask import Flask, render_template, request, jsonify
import openai

app = Flask(__name__)

# Set OpenAI API key (preferably through environment variable)


# Command type validation
def validate_command_type(command_type):
    if command_type not in ["grammar", "translate", "antonym-synonym"]:
        raise ValueError("Invalid command type")


# Error handling
def handle_error(error):
    return f"An error occurred: {error}"


# Function to interact with OpenAI API
def chat_with_bot(prompt, command_type):
    if command_type == "antonym-synonym":
        if not prompt:  # Check if user provided input
            return "What sentence or word do you want to inquire about?"
        else:
            try:
                # Retrieve synonyms and antonyms using OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": f"You are now using the {command_type} command."},
                              {"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return response.choices[0].message['content'].strip()
            except Exception as e:
                return f"An error occurred: {str(e)}"
    elif command_type == "translate":
        if not prompt:  # Check if user provided input
            return "What sentence or word do you want to translate?"
        else:
            try:
                # Perform translation using OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": f"You are now using the {command_type} command."},
                              {"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return response.choices[0].message['content'].strip()
            except Exception as e:
                return f"An error occurred: {str(e)}"
    elif command_type == "grammar":
        if not prompt:  # Check if user provided input
            return "What sentence or word do you want to check for grammar?"
        else:
            try:
                # Perform grammar check using OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": f"You are now using the {command_type} command."},
                              {"role": "user", "content": prompt}],
                    max_tokens=100
                )
                return response.choices[0].message['content'].strip()
            except Exception as e:
                return f"An error occurred: {str(e)}"
    else:
        try:
            # Handle other command types
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": f"You are now using the {command_type} command."},
                          {"role": "user", "content": prompt}],
                max_tokens=100
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"An error occurred: {str(e)}"





# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    command_type = request.form['command_type']

    try:
        validate_command_type(command_type)
        bot_response = chat_with_bot(user_message, command_type)
        return jsonify({'bot_response': bot_response})
    except ValueError as ve:
        return jsonify({'bot_response': handle_error(str(ve))})


if __name__ == '__main__':
    app.run(debug=True)
