# main.py
from experiments import exp2_with_feedback
from experiments.exp1 import exp1_no_feedback

if __name__ == "__main__":
    print("选择实验：1=无反馈，2=有反馈，3=网络拓扑效应")
    choice = input("请输入实验编号：")
    if choice == "1":
        exp1_no_feedback.run_exp1()
    elif choice == "2":
        exp2_with_feedback.run_exp2()
    elif choice == "3":
        exp3_network_effect.run_exp3()
