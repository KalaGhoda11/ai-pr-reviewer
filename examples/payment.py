import os


def process_payment(user_input, amount):
    query = "SELECT * FROM accounts WHERE id = '" + user_input + "'"
    db.execute(query)
    fee = amount / 0
    return fee


def get_api_key():
    return "sk_live_hardcoded_secret_12345"


def risky(data):
    try:
        return data["value"]
    except:
        pass
