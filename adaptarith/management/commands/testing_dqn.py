import os

import torch
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from dqn_model.dqn import DQN

import pandas as pd


def categorize_score(score):
    for idx, x in enumerate(range(50, 91, 15)):
        if score < x:
            return idx
    return idx + 1

def normalise_sequence(learner_sequence, max_sequence_length):
    padded_arr = learner_sequence + [-1] * (max_sequence_length - len(learner_sequence) -1)
    for idx, x in enumerate(padded_arr):
        if idx % 2 == 0:
            if x == -1:
                padded_arr[idx] = 0
            else:
                padded_arr[idx] = x /320
        elif x != -1:
            padded_arr[idx] = x / 6
    return padded_arr

def get_true_next_score(index, current_user_data):
    try:
        next_score = categorize_score(current_user_data.iloc[index].score)
    except IndexError:
        next_score = -1

    try:
        next_activities = current_user_data.iloc[index+1].total_vle_before_assessment
    except IndexError:
        next_activities = 0
    return next_score, next_activities

def get_predicted_next_score(model, learner_sequence):
    state_tensor = torch.tensor(learner_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():  # Disable gradient calculation since we are only inferring
        q_values = model(state_tensor)  # Get Q-values from the model

    # Select the action with the highest Q-value (for DQN)
    if q_values.dim() == 1:
        q_values = q_values.unsqueeze(0)
    #print("Q-values:", q_values.numpy())
    action = torch.argmax(q_values, dim=1).item()

    return action

class Command(BaseCommand):
    help = _(u"For testing the model via command line")
    errors = []

    def handle(self, *args, **options):
        state_dict_path = os.path.join(settings.BASE_DIR, 'dqn_model', 'results', "20241217200224-250k", "model.pth")

        data_file = os.path.join(settings.BASE_DIR, 'data', 'ou', "studentassessment_course_bbb2013j_with_activities.csv")

        # load model and DQN
        model = DQN(input_size=21,
                         hidden_dims=128,
                         output_size=7)
        model.load_state_dict(torch.load(state_dict_path, weights_only=False))
        model.eval()

        # load data, split into users
        activity = pd.read_csv(data_file)
        all_users = activity['id_student'].unique()

        num_exact_correct = 0
        num_exact_incorrect = 0

        num_close_correct = 0
        num_close_incorrect = 0

        # loop through users until no more activity
        for user in all_users:
            current_user_data = activity.loc[activity['id_student'] == user].sort_values(by='date_submitted')
            first_activities = current_user_data.iloc[0].total_vle_before_assessment
            first_score = categorize_score(current_user_data.iloc[0].score)
            try:
                second_activities = current_user_data.iloc[1].total_vle_before_assessment
            except IndexError:
                second_activities = 0
            learner_sequence = [first_activities, first_score, second_activities]

            for i in range(2,11):
                normalised_sequence = normalise_sequence(learner_sequence,22)
                print(normalised_sequence)

                # check if suggested action matches the actual one
                actual_next_score_category, next_activities_count = get_true_next_score(i, current_user_data)
                print(actual_next_score_category)

                predicted_next_score = get_predicted_next_score(model, normalised_sequence)
                print(predicted_next_score-1)

                if actual_next_score_category == -1:
                    if predicted_next_score == 0:
                        num_exact_correct += 1
                    else:
                        num_exact_incorrect += 1
                    if abs(predicted_next_score-1) < 2:
                        num_close_correct += 1
                    else:
                        num_close_incorrect += 1
                    break

                learner_sequence.append(actual_next_score_category)
                learner_sequence.append(next_activities_count)

                if actual_next_score_category == predicted_next_score-1:
                    num_exact_correct += 1
                else:
                    num_exact_incorrect += 1
                if abs(actual_next_score_category - (predicted_next_score-1)) <= 1:
                    num_close_correct += 1
                else:
                    num_close_incorrect += 1

        expected_from_random = 100/7
        actual_exact = num_exact_correct*100 / (num_exact_correct+num_exact_incorrect)
        actual_close = num_close_correct * 100 / (num_close_correct + num_close_incorrect)
        print(f"random {expected_from_random:.2f}, actual exact: {actual_exact:.2f}, actual close {actual_close:.2f}")
        print(f"Exactly num correct {num_exact_correct}, num incorrect: {num_exact_incorrect}")
        print(f"Close(+/-1) num correct {num_close_correct}, num incorrect: {num_close_incorrect}")