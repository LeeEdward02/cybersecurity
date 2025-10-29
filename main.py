# main.py
from experiments.exp1 import exp1_no_feedback
from experiments.exp2 import exp2_with_feedback, exp2_with_feedback_fig6
from experiments.exp3 import exp3_fig7, exp3_fig8, exp_fig9, exp_fig10, exp3_table3

if __name__ == "__main__":
    print("选择实验：1=无反馈，2=有反馈，3=网络拓扑效应")
    choice = input("请输入实验编号：")
    if choice == "1":
        exp1_no_feedback.run_exp1()
    elif choice == "2":
        exp2_with_feedback.run_exp2()
        exp2_with_feedback_fig6.run_exp2_fig6()
    elif choice == "3":
        exp3_fig7.main()
        exp3_fig8.reproduce_fig8()
        exp_fig9.reproduce_fig9()
        exp_fig10.main()
        exp3_table3.main()
