from student_expert_flow.config import load_config


def test_load_configs():
    print("Testing config loading...")
    try:
        expert_config = load_config('configs/expert_config.yaml', 'expert')
        print("Expert config loaded successfully:")
        print(expert_config.model_dump_json(indent=2))

        student_config = load_config('configs/student_config.yaml', 'student')
        print("\nStudent config loaded successfully:")
        print(student_config.model_dump_json(indent=2))
        print("\nConfig loading test passed.")
        return True
    except Exception as e:
        print(f"\nConfig loading test failed: {e}")
        return False


if __name__ == "__main__":
    test_load_configs()
