"""Authentication and access control mechanisms for Helios."""

import logging
import os
import json
import time
import hashlib
import secrets
import base64
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum, auto
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for access control."""
    ADMIN = auto()
    OPERATOR = auto()
    ANALYST = auto()
    VIEWER = auto()


class ResourceType(Enum):
    """Types of resources that can be protected."""
    HARDWARE = auto()
    SIMULATION = auto()
    NETWORK = auto()
    COGNITIVE = auto()
    FPGA = auto()
    SYSTEM = auto()


@dataclass
class User:
    """User information for authentication."""
    username: str
    password_hash: str
    salt: str
    role: UserRole
    last_login: Optional[float] = None
    token: Optional[str] = None
    token_expiry: Optional[float] = None


class AuthManager:
    """
    Authentication and access control manager for Helios.
    Handles user authentication and permission checks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the authentication manager."""
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, str] = {}  # token -> username
        self.token_lifetime = 3600  # 1 hour in seconds
        
        # Default permissions by role
        self.role_permissions: Dict[UserRole, Dict[ResourceType, List[str]]] = {
            UserRole.ADMIN: {
                resource_type: ["read", "write", "execute", "configure"] 
                for resource_type in ResourceType
            },
            UserRole.OPERATOR: {
                ResourceType.HARDWARE: ["read", "write", "execute"],
                ResourceType.SIMULATION: ["read", "write", "execute"],
                ResourceType.NETWORK: ["read", "write"],
                ResourceType.COGNITIVE: ["read", "execute"],
                ResourceType.FPGA: ["read", "execute"],
                ResourceType.SYSTEM: ["read"]
            },
            UserRole.ANALYST: {
                ResourceType.HARDWARE: ["read"],
                ResourceType.SIMULATION: ["read", "write"],
                ResourceType.NETWORK: ["read"],
                ResourceType.COGNITIVE: ["read", "write"],
                ResourceType.FPGA: ["read"],
                ResourceType.SYSTEM: ["read"]
            },
            UserRole.VIEWER: {
                resource_type: ["read"] for resource_type in ResourceType
            }
        }
        
        # Load users from config if provided
        if config_path:
            self.load_users(config_path)
        else:
            # Create default admin user if no config
            self._create_default_admin()
    
    def _create_default_admin(self) -> None:
        """Create a default admin user."""
        if not self.users:
            # Generate a random password for the admin
            password = secrets.token_urlsafe(12)
            salt = secrets.token_hex(16)
            password_hash = self._hash_password("admin123", salt)
            
            self.users["admin"] = User(
                username="admin",
                password_hash=password_hash,
                salt=salt,
                role=UserRole.ADMIN
            )
            
            logger.warning(f"Created default admin user with password: admin123")
            logger.warning("Please change this password immediately after first login")
    
    def load_users(self, config_path: str) -> bool:
        """
        Load users from a configuration file.
        
        Args:
            config_path: Path to the user configuration file
            
        Returns:
            Success status
        """
        try:
            if not os.path.exists(config_path):
                logger.error(f"User configuration file not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                user_data = json.load(f)
            
            for username, data in user_data.items():
                try:
                    role = UserRole[data.get("role", "VIEWER")]
                    self.users[username] = User(
                        username=username,
                        password_hash=data["password_hash"],
                        salt=data["salt"],
                        role=role,
                        last_login=data.get("last_login")
                    )
                except (KeyError, ValueError) as e:
                    logger.error(f"Invalid user data for {username}: {e}")
            
            logger.info(f"Loaded {len(self.users)} users from configuration")
            return True
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
            return False
    
    def save_users(self, config_path: str) -> bool:
        """
        Save users to a configuration file.
        
        Args:
            config_path: Path to save the user configuration
            
        Returns:
            Success status
        """
        try:
            user_data = {}
            for username, user in self.users.items():
                user_data[username] = {
                    "password_hash": user.password_hash,
                    "salt": user.salt,
                    "role": user.role.name,
                    "last_login": user.last_login
                }
            
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(user_data, f, indent=2)
            
            logger.info(f"Saved {len(self.users)} users to configuration")
            return True
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
            return False
    
    def _hash_password(self, password: str, salt: str) -> str:
        """
        Hash a password with the given salt.
        
        Args:
            password: Plain text password
            salt: Salt for the hash
            
        Returns:
            Password hash
        """
        password_bytes = password.encode('utf-8')
        salt_bytes = bytes.fromhex(salt)
        hash_bytes = hashlib.pbkdf2_hmac('sha256', password_bytes, salt_bytes, 100000)
        return hash_bytes.hex()
    
    def _generate_token(self) -> str:
        """
        Generate a secure authentication token.
        
        Returns:
            Authentication token
        """
        return secrets.token_urlsafe(32)
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate a user and return a session token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Session token if authentication successful, None otherwise
        """
        if username not in self.users:
            logger.warning(f"Authentication failed: User {username} not found")
            return None
        
        user = self.users[username]
        password_hash = self._hash_password(password, user.salt)
        
        if password_hash != user.password_hash:
            logger.warning(f"Authentication failed: Invalid password for {username}")
            return None
        
        # Generate a new token
        token = self._generate_token()
        expiry = time.time() + self.token_lifetime
        
        # Update user information
        user.last_login = time.time()
        user.token = token
        user.token_expiry = expiry
        
        # Store session
        self.sessions[token] = username
        
        logger.info(f"User {username} authenticated successfully")
        return token
    
    def validate_token(self, token: str) -> Optional[str]:
        """
        Validate a session token.
        
        Args:
            token: Session token
            
        Returns:
            Username if token is valid, None otherwise
        """
        if token not in self.sessions:
            return None
        
        username = self.sessions[token]
        user = self.users[username]
        
        # Check if token has expired
        if not user.token_expiry or time.time() > user.token_expiry:
            # Remove expired token
            del self.sessions[token]
            user.token = None
            user.token_expiry = None
            return None
        
        return username
    
    def logout(self, token: str) -> bool:
        """
        Log out a user by invalidating their session token.
        
        Args:
            token: Session token
            
        Returns:
            Success status
        """
        if token not in self.sessions:
            return False
        
        username = self.sessions[token]
        user = self.users[username]
        
        # Clear token
        del self.sessions[token]
        user.token = None
        user.token_expiry = None
        
        logger.info(f"User {username} logged out")
        return True
    
    def check_permission(self, token: str, resource_type: ResourceType, action: str) -> bool:
        """
        Check if a user has permission to perform an action on a resource.
        
        Args:
            token: Session token
            resource_type: Type of resource
            action: Action to perform (read, write, execute, configure)
            
        Returns:
            True if user has permission, False otherwise
        """
        username = self.validate_token(token)
        if not username:
            logger.warning("Permission check failed: Invalid or expired token")
            return False
        
        user = self.users[username]
        
        # Check if user's role has permission for this action on this resource
        if resource_type in self.role_permissions[user.role]:
            if action in self.role_permissions[user.role][resource_type]:
                return True
        
        logger.warning(f"Permission denied: {username} cannot {action} {resource_type.name}")
        return False
    
    def create_user(self, admin_token: str, username: str, password: str, role: UserRole) -> bool:
        """
        Create a new user (requires admin privileges).
        
        Args:
            admin_token: Admin session token
            username: New username
            password: New password
            role: User role
            
        Returns:
            Success status
        """
        # Check admin permissions
        admin_username = self.validate_token(admin_token)
        if not admin_username:
            logger.warning("Create user failed: Invalid or expired admin token")
            return False
        
        admin = self.users[admin_username]
        if admin.role != UserRole.ADMIN:
            logger.warning(f"Create user failed: {admin_username} is not an admin")
            return False
        
        # Check if username already exists
        if username in self.users:
            logger.warning(f"Create user failed: Username {username} already exists")
            return False
        
        # Create new user
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        
        self.users[username] = User(
            username=username,
            password_hash=password_hash,
            salt=salt,
            role=role
        )
        
        logger.info(f"User {username} created with role {role.name}")
        return True
    
    def change_password(self, token: str, old_password: str, new_password: str) -> bool:
        """
        Change a user's password.
        
        Args:
            token: Session token
            old_password: Current password
            new_password: New password
            
        Returns:
            Success status
        """
        username = self.validate_token(token)
        if not username:
            logger.warning("Change password failed: Invalid or expired token")
            return False
        
        user = self.users[username]
        
        # Verify old password
        old_hash = self._hash_password(old_password, user.salt)
        if old_hash != user.password_hash:
            logger.warning(f"Change password failed: Invalid old password for {username}")
            return False
        
        # Update password
        user.salt = secrets.token_hex(16)
        user.password_hash = self._hash_password(new_password, user.salt)
        
        logger.info(f"Password changed for user {username}")
        return True