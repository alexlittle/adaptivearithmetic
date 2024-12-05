
def format_question(question):
    if question.topic == 'add':
        return f"{question.first_term} + {question.second_term}"
    if question.topic == 'subtract':
        return f"{question.first_term} - {question.second_term}"
    if question.topic == 'multiply':
        return f"{question.first_term} * {question.second_term}"
    if question.topic == 'divide':
        return f"{question.first_term} / {question.second_term}"


def get_next_question():
    pass