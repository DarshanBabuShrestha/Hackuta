"""Guided capture of real Arduino wrist motion for future analysis/training.

Examples:
  python -m scripts.record_wrist_movements --port COM9
  python -m scripts.record_wrist_movements --port COM9 --label confirmation
"""
import argparse
import csv
from datetime import datetime, timezone
import time
from pathlib import Path

from backend.serial_bridge import parse_message


NORMAL_MOVEMENTS = [
    'Rotate the wrist inward, then return to neutral',
    'Rotate the wrist outward, then return to neutral',
    'Wave the hand side to side',
    'Point forward, then lower the hand',
    'Pick up an imaginary light object and set it down',
    'Turn the palm up, then palm down',
    'Make a small natural conversational hand gesture',
    'Reach forward and return the hand to rest',
    'Make a gentle wrist circle clockwise',
    'Make a gentle wrist circle counterclockwise',
    'Raise the hand as if greeting someone',
    'Lower the hand from a raised position',
    'Point left, then return to neutral',
    'Point right, then return to neutral',
    'Pretend to tap a button twice',
    'Pretend to turn a small door handle',
    'Move the hand as if writing a short note',
    'Brush a small crumb from a table',
    'Reach to the side and return to rest',
    'Flip the palm over once',
    'Make a gentle open-hand wave',
    'Make a small natural counting gesture',
    'Pretend to pick up a cup and set it down',
    'Move the wrist gently up and down',
    'Move the wrist gently left and right',
    'Bring the hand toward the chest and return it',
    'Make a small pointing gesture downward',
    'Make a small pointing gesture upward',
    'Pretend to adjust a sleeve cuff',
    'Rest the hand after one ordinary conversational gesture',
]

FIELDS = ('timestamp_ms', 'captured_at_utc', 'label', 'movement_id', 'instruction',
          'ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps')


def drain_for(device, seconds):
    """Keep the serial RX buffer current without saving instruction/pause data."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        device.readline()


def capture(args):
    try:
        import serial
    except ImportError as exc:
        raise SystemExit('pyserial is required; install the project requirements first.') from exc

    label = ('NORMAL_MOVEMENT' if args.label == 'normal'
             else 'CONFIRMATION_GESTURE')
    instructions = (NORMAL_MOVEMENTS if args.label == 'normal' else
                    ['Perform two deliberate opposite-direction wrist twists, then hold still for one second'] * args.count)
    destination = args.output or Path('data') / f'wrist_{args.label}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f'Connecting to {args.port} at {args.baud} baud. Close Arduino Serial Monitor first.')
    with serial.Serial(args.port, args.baud, timeout=0.25) as device, destination.open(
            'w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        device.reset_input_buffer()
        movement_count = min(args.count, len(instructions))
        for index in range(movement_count):
            instruction = instructions[index]
            print(f'\nMovement {index + 1}/{movement_count}: {instruction}')
            for value in (3, 2, 1):
                print(f'  Begin in {value}...', flush=True)
                drain_for(device, 1)
            deadline = time.monotonic() + args.record_seconds
            rows = 0
            while time.monotonic() < deadline:
                raw = device.readline()
                if not raw:
                    continue
                try:
                    sample = parse_message(raw)
                except (ValueError, UnicodeError):
                    continue
                ax, ay, az, gx, gy, gz = sample.values
                writer.writerow({
                    'timestamp_ms': sample.timestamp_ms,
                    'captured_at_utc': datetime.now(timezone.utc).isoformat(),
                    'label': label,
                    'movement_id': index + 1,
                    'instruction': instruction,
                    'ax_g': ax, 'ay_g': ay, 'az_g': az,
                    'gx_dps': gx, 'gy_dps': gy, 'gz_dps': gz,
                })
                rows += 1
            output.flush()
            print(f'  Saved {rows} real sensor samples.')
            if index + 1 < movement_count:
                print('  Pause for one second before the next movement.')
                drain_for(device, 1)
    print(f'\nSaved {label} recording to {destination.resolve()}')
    print('No samples are synthesized; empty or disconnected intervals are not filled in.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Record real Odyssey Guardian wrist IMU movements')
    parser.add_argument('--port', required=True, help='Arduino serial port, e.g. COM9')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--label', choices=('normal', 'confirmation'), default='normal')
    parser.add_argument('--count', type=int, default=30,
                        help='Number of confirmation gesture takes (normal mode guides up to 30)')
    parser.add_argument('--record-seconds', type=float, default=4.0,
                        help='Real sensor capture duration for each prompted movement')
    parser.add_argument('--output', type=Path, help='CSV output path')
    args = parser.parse_args(argv)
    if args.count < 1 or args.record_seconds <= 0:
        parser.error('--count and --record-seconds must be positive')
    capture(args)


if __name__ == '__main__':
    main()
