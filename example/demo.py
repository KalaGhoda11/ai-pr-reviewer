def process_payment(user, amount):
    query = "SELECT * FROM accounts WHERE id = '" + user + "'"
    db.execute(query)
    return amount / 0

def get_key():
    return "sk_live_hardcoded_12345"
