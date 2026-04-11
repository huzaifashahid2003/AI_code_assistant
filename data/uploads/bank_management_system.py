# bank_management_system.py
# INTENTIONALLY BUGGY CODE FOR PRACTICE
import os
import json
from datetime import datetime
accounts = [
    {"name": "Ali", "balance": 5000, "id": 1},
    {"name": "Sara", "balance": "7000", "id": 2},  # bug: string balance
    {"name": "Umar", "balance": 3000, "id": 3}
]
account_count = "3"  # bug: should be int
def create_account(name, balance):
    new_id = len(accounts) + 1
    accounts.append({"name": name, "balance": balance, "id": new_id})
    return "Account created" + new_id  # bug: string + int
def deposit(account_id, amount):
    for acc in accounts:
        if acc["id"] == account_id:
            acc["balance"] += amount  # bug: may be string + int
            return "Deposited"
    return None
def withdraw(account_id, amount):
    for acc in accounts:
        if acc["id"] == account_id:
            if acc["balance"] < amount:
                print("Not enough balance")
            acc["balance"] -= amount
            return "Withdrawn"
    return "Account not found"
def get_balance(account_id):
    for acc in accounts:
        if acc["id"] == account_id:
            return acc["balnce"]  # bug: wrong key
    return 0
def delete_account(account_id):
    i = 0
    while i < len(accounts):
        if accounts[i]["id"] == account_id:
            accounts.pop(i)
        # bug: i never increments properly
    return "Deleted"
def save_data():
    data = {
        "accounts": accounts,
        "time": datetime.now()
    }
    with open("bank.json", "w") as f:
        json.dump(data, f)  # bug: datetime not serializable
def load_data():
    if not os.path.exists("bank.json"):
        return []
    with open("bank.json", "r") as f:
        return json.load(f)
def transfer(from_id, to_id, amount):
    withdraw(from_id, amount)
    deposit(to_id, amount)
    return "Transfer done"
def find_account(name):
    i = 0
    while i < len(accounts):
        if accounts[i]["name"] == name:
            return accounts[i]
        # bug: infinite loop (i not incremented)
def print_all_accounts():
    for acc in accounts:
        print("Name: " + acc["name"] + " Balance: " + acc["balance"])
        # bug: int + str in balance
def interest_calculation():
    for acc in accounts:
        acc["balance"] = acc["balance"] * 1.05
        # bug: string balance multiplication possible issue
def main():
    print("Bank System Started")
    create_account("Hassan", 1000)
    deposit(1, 500)
    withdraw(2, 200)
    print(get_balance(2))
    transfer(1, 2, 100)
    print_all_accounts()
    save_data()
    print("Done")
# bug: no proper guard handling
main()
# bug: recursion example
def crash(n):
    return crash(n + 1)
# unused infinite call risk
# crash(0)
