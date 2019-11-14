"""

Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""
import csv

def main():
    """
    """
    # Get data
    with open('datasets/animal.csv', 'r') as f:
        reader = csv.reader(f)
        animal_mouse_data = [r for r in reader if r][1:]

    with open('datasets/computer.csv', 'r') as f:
        reader = csv.reader(f)
        computer_mouse_data = [r for r in reader if r][1:]

    return


if __name__ == '__main__':
    main()
    exit(0)
