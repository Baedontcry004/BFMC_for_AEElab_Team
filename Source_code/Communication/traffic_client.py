import socket
import json
import time
import threading
from typing import List

# Import the utility functions (Ensure keyDealer has the encrypt_data function we added)
from utils.keyDealer import load_public_key, encrypt_data

class TrafficClient(threading.Thread):
    """
    TCP Client for V2X Traffic Communication.
    Handles connection to the Bosch Server, subscription, and periodic telemetry reporting.
    """

    def __init__(self, server_ip: str, server_port: int, car_id: int, shared_state: dict, pub_key_path: str):
        super().__init__()
        self.server_ip = server_ip
        self.server_port = server_port
        self.car_id = car_id
        self.shared_state = shared_state
        self.running = True
        self.sock = None
        
        # Load the RSA Public Key from the specified path
        self.public_key = load_public_key(pub_key_path)
        print(f"[TCP_CLIENT] Initialized. Target Server: {self.server_ip}:{self.server_port}")

    def connect_to_server(self) -> bool:
        """
        Establish a TCP socket connection with the server. Includes a basic retry mechanism.
        """
        while self.running:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0)
                self.sock.connect((self.server_ip, self.server_port))
                print(f"[TCP_CLIENT] Successfully connected to {self.server_ip}:{self.server_port}")
                return True
            except socket.error as e:
                print(f"[TCP_CLIENT] Connection failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)
        return False

    def send_encrypted_payload(self, payload_dict: dict):
        """
        Convert dictionary to JSON string, encrypt via RSA, and transmit over TCP.
        """
        try:
            json_str = json.dumps(payload_dict)
            encrypted_bytes = encrypt_data(self.public_key, json_str.encode('utf-8'))
            self.sock.sendall(encrypted_bytes)
        except Exception as e:
            print(f"[TCP_CLIENT] Error during transmission: {e}")

    def subscribe_to_location(self):
        """
        Send the initial mandatory subscription packet to receive server recognition and GPS data.
        """
        sub_payload = {
            "reqORinfo": "info",
            "type": "locIDsub",
            "locID": self.car_id,
            "freq": 0.5
        }
        self.send_encrypted_payload(sub_payload)
        print(f"[TCP_CLIENT] Subscription packet sent for Car ID: {self.car_id}")

    def create_telemetry_payload(self, cmd_type: str, values: List[float]) -> dict:
        """
        Generate the standard Bosch V2X telemetry format.
        """
        payload = {
            "reqORinfo": "info",
            "type": cmd_type
        }
        if len(values) > 0:
            payload["value1"] = float(values[0])
        if len(values) > 1:
            payload["value2"] = float(values[1])
        if len(values) > 2:
            payload["value3"] = float(values[2])
            
        return payload

    def run(self):
        """
        Main thread execution loop.
        """
        if not self.connect_to_server():
            return

        # Crucial step: Register the car with the server immediately after connecting
        self.subscribe_to_location()

        # Enter the periodic reporting loop
        while self.running:
            try:
                # 1. Fetch current data from the shared state (assuming it's updated by other logic)
                current_speed = self.shared_state.get("speed", 0.0)
                current_pos_x = self.shared_state.get("pos_x", 0.0)
                current_pos_y = self.shared_state.get("pos_y", 0.0)

                # 2. Package and send Speed
                speed_payload = self.create_telemetry_payload("deviceSpeed", [current_speed])
                self.send_encrypted_payload(speed_payload)

                # 3. Package and send Position
                pos_payload = self.create_telemetry_payload("devicePos", [current_pos_x, current_pos_y])
                self.send_encrypted_payload(pos_payload)

                # Sleep to maintain the reporting frequency (e.g., 2Hz / 0.5s)
                time.sleep(0.5)

            except Exception as e:
                print(f"[TCP_CLIENT] Critical error in reporting loop: {e}. Attempting reconnect...")
                self.connect_to_server()
                self.subscribe_to_location()

    def stop(self):
        """
        Gracefully terminate the thread and close the socket.
        """
        self.running = False
        if self.sock:
            self.sock.close()
        print("[TCP_CLIENT] Terminated.")

# ==============================================================================
# PHẦN CHẠY THỬ NGHIỆM ĐỘC LẬP (STANDALONE TESTING BLOCK)
# ==============================================================================
if __name__ == "__main__":
    # 1. Tạo một cái kho dữ liệu giả lập (Mock Shared State)
    mock_shared_state = {
        "speed": 15.5,
        "pos_x": 1.2,
        "pos_y": 3.4
    }

    # 2. CẤU HÌNH IP VÀ KHÓA (CONFIGURATION)
    # ⚠️ THAY SERVER_IP BẰNG IP CỦA LAPTOP ĐANG CHẠY MÁY CHỦ BOSCH!
    SERVER_IP = "192.168.31.87" 
    SERVER_PORT = 5000
    CAR_ID = 3
    
    # ⚠️ ĐẢM BẢO ĐƯỜNG DẪN ĐẾN FILE KEY LÀ ĐÚNG VỚI CẤU TRÚC THƯ MỤC!
    KEY_PATH = "keys/publickey_server_test.pem" 

    # 3. Khởi tạo và bấm nút CHẠY!
    print("[TEST] Bắt đầu khởi động TrafficClient...")
    client = TrafficClient(SERVER_IP, SERVER_PORT, CAR_ID, mock_shared_state, KEY_PATH)
    client.start() # Bấm nút chạy thread!

    try:
        # Giữ cho chương trình chính (Main thread) không bị tắt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[TEST] Người dùng vừa bấm Ctrl+C. Đang tắt Client...")
        client.stop()
        client.join()