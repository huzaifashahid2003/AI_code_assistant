import os
import json
import math
from datetime import datetime

# ==========================================
# STUDENT MANAGEMENT SYSTEM - FULL OF BUGS
# ==========================================

# Bug 1: Wrong variable type
student_count = "10"  # String hai, int hona chahiye tha

# Bug 2: List with mixed types
students = [
    {"name": "Ali", "age": 20, "grade": 85},
    {"name": "Sara", "age": "22", "grade": 90},  # age string hai
    {"name": "Umar", "age": 19, "grade": None},  # grade None hai
]

# Bug 3: Division by zero possible
def calculate_average(total, count):
    return total / count  # Agar count=0 ho toh ZeroDivisionError

# Bug 4: Wrong indentation logic
def get_grade_letter(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
       return "D"    # Indentation galat hai
    else:
        return "F"

# Bug 5: KeyError - wrong key name
def print_student_info(student):
    print(f"Name: {student['name']}")
    print(f"Age: {student['age']}")
    print(f"Score: {student['score']}")  # 'score' key exist nahi karti, 'grade' hai

# Bug 6: TypeError - string + int
def display_summary():
    print("Total Students: " + student_count + 5)  # String aur int add nahi ho sakta

# Bug 7: AttributeError
def process_name(name):
    return name.upper().strip().reverse()  # reverse() strings ka method nahi hai

# Bug 8: Index out of range
def get_top_student():
    sorted_students = sorted(students, key=lambda x: x['grade'] or 0)
    return sorted_students[10]  # Sirf 3 students hain, index 10 exist nahi karta

# Bug 9: FileNotFoundError
def load_data_from_file():
    with open("students_data.txt", "r") as f:  # File exist nahi karti
        data = f.read()
    return data

# Bug 10: Infinite loop
def find_student_by_name(name):
    i = 0
    while i < len(students):  # i kabhi increment nahi hota — infinite loop!
        if students[i]['name'] == name:
            return students[i]

# Bug 11: Wrong math operation
def calculate_percentage(obtained, total):
    percentage = obtained / total * 10  # 10 nahi, 100 hona chahiye tha
    return percentage

# Bug 12: None comparison error
def check_passing(student):
    if student['grade'] > 50:  # None > 50 TypeError dega
        return "Pass"
    return "Fail"

# Bug 13: Datetime format galat
def get_current_date():
    now = datetime.now()
    return now.strftime("%Y-%DD-%MM")  # %DD aur %MM galat format codes hain

# Bug 14: JSON dump error
def save_students_to_json():
    data = {
        "students": students,
        "date": datetime.now()  # datetime object JSON serializable nahi hota
    }
    with open("output.json", "w") as f:
        json.dump(data, f)

# Bug 15: Recursive function without base case
def factorial(n):
    return n * factorial(n - 1)  # Base case nahi hai — RecursionError!

# ==========================================
# MAIN PROGRAM - SAB BUGS TRIGGER HONGE
# ==========================================

if __name__ == "__main__":
    print("=== Student Management System ===")
    
    # Bug 6 trigger
    display_summary()
    
    # Bug 5 trigger
    print_student_info(students[0])
    
    # Bug 3 trigger
    avg = calculate_average(0, 0)
    print(f"Average: {avg}")
    
    # Bug 7 trigger
    name = process_name("ali")
    
    # Bug 8 trigger
    top = get_top_student()
    
    # Bug 9 trigger
    load_data_from_file()
    
    # Bug 12 trigger
    for student in students:
        result = check_passing(student)
        print(f"{student['name']}: {result}")
    
    # Bug 15 trigger
    print(factorial(5))
    
    print("=== Program Complete ===")
