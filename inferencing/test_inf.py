import argparse
from dummyMain import test_pre_processed


def main():
    parser = argparse.ArgumentParser(description="Run test_pre_processed on three images.")
    parser.add_argument("--anchor", required=True, help="Path to anchor image")
    parser.add_argument("--negative", required=True, help="Path to negative image")
    parser.add_argument("--positive", required=True, help="Path to positive image")

    args = parser.parse_args()

    print("Running test_pre_processed with:")
    print(f"  Anchor:   {args.anchor}")
    print(f"  Negative: {args.negative}")
    print(f"  Positive: {args.positive}")

    try:
        result = test_pre_processed(
            control_image_path=args.anchor,
            clean_b_path=args.negative,
            dirty_b_path=args.positive
        )
        print("\n✅ Test completed successfully.")
        print("Result:", result)
    except Exception as e:
        print("\n❌ Error running test_pre_processed:", e)


if __name__ == "__main__":
    main()
