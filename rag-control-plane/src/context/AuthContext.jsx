import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const DEMO_USERS = {
  ADMIN: {
    id: 'usr-admin-01',
    username: 'dr.sarah',
    name: 'Dr. Sarah Lin',
    title: 'Chief Medical Officer & Knowledge Custodian',
    role: 'ADMIN',
    department: 'Clinical Governance & AASLD Consortium',
    avatar: 'SL',
    badgeColor: 'indigo'
  },
  TECHNICIAN: {
    id: 'usr-tech-01',
    username: 'alex.rivera',
    name: 'Alex Rivera',
    title: 'Lead RAG Infrastructure & MLOps Engineer',
    role: 'TECHNICIAN',
    department: 'Applied AI & Vector Systems Engineering',
    avatar: 'AR',
    badgeColor: 'cyan'
  }
};

export function AuthProvider({ children }) {
  // Default to ADMIN for instant demo exploration, or check localStorage
  const [currentUser, setCurrentUser] = useState(() => {
    const saved = localStorage.getItem('rag_control_role');
    if (saved === 'TECHNICIAN') return DEMO_USERS.TECHNICIAN;
    return DEMO_USERS.ADMIN;
  });

  const [currentPath, setCurrentPath] = useState(() => {
    const saved = localStorage.getItem('rag_control_role');
    return saved === 'TECHNICIAN' ? '/technician' : '/admin';
  });

  const [accessDeniedMessage, setAccessDeniedMessage] = useState(null);

  // Sync role to localStorage
  useEffect(() => {
    if (currentUser) {
      localStorage.setItem('rag_control_role', currentUser.role);
    }
  }, [currentUser]);

  // Login method supporting both role-selection and credentials
  const login = (roleSelected, credentials = {}) => {
    const roleKey = (roleSelected || 'ADMIN').toUpperCase();
    const targetUser = DEMO_USERS[roleKey] || DEMO_USERS.ADMIN;
    setCurrentUser(targetUser);
    const targetPath = roleKey === 'TECHNICIAN' ? '/technician' : '/admin';
    setCurrentPath(targetPath);
    setAccessDeniedMessage(null);
  };

  const logout = () => {
    setCurrentUser(null);
    setCurrentPath('/login');
    localStorage.removeItem('rag_control_role');
  };

  // Quick Demo Role Switcher (accessible from top navbar)
  const switchRole = (newRole) => {
    const roleKey = (newRole || '').toUpperCase();
    if (roleKey === 'TECHNICIAN') {
      setCurrentUser(DEMO_USERS.TECHNICIAN);
      setCurrentPath('/technician');
    } else {
      setCurrentUser(DEMO_USERS.ADMIN);
      setCurrentPath('/admin');
    }
    setAccessDeniedMessage(null);
  };

  // Route Navigator with Strict Protection
  const navigateTo = (path) => {
    if (!currentUser) {
      setCurrentPath('/login');
      return;
    }

    if (path.startsWith('/technician') && currentUser.role !== 'TECHNICIAN') {
      setAccessDeniedMessage({
        attemptedPath: path,
        requiredRole: 'TECHNICIAN',
        currentRole: currentUser.role,
        message: 'Access Denied: Administrative accounts do not have permission to modify vector embedding math or raw chunking configurations. Please switch to the Technician role to access this telemetry.'
      });
      return;
    }

    if (path.startsWith('/admin') && currentUser.role !== 'ADMIN') {
      setAccessDeniedMessage({
        attemptedPath: path,
        requiredRole: 'ADMIN',
        currentRole: currentUser.role,
        message: 'Access Denied: Technical engineering accounts cannot publish or unpublish clinical governance documents directly. Please switch to the Clinical Admin role.'
      });
      return;
    }

    setAccessDeniedMessage(null);
    setCurrentPath(path);
  };

  const clearAccessDenied = () => {
    setAccessDeniedMessage(null);
  };

  return (
    <AuthContext.Provider
      value={{
        currentUser,
        currentPath,
        login,
        logout,
        switchRole,
        navigateTo,
        accessDeniedMessage,
        clearAccessDenied,
        isAdmin: currentUser?.role === 'ADMIN',
        isTechnician: currentUser?.role === 'TECHNICIAN',
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

