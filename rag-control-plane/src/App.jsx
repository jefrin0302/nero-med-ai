import React from 'react';
import { useAuth } from './context/AuthContext';
import Navbar from './components/common/Navbar';
import AccessDeniedModal from './components/common/AccessDeniedModal';
import LoginPage from './components/auth/LoginPage';
import AdminDashboard from './components/admin/AdminDashboard';
import TechnicianDashboard from './components/technician/TechnicianDashboard';

export default function App() {
  const { currentUser, currentPath } = useAuth();

  // If not logged in or explicitly at login route, show unified login
  if (!currentUser || currentPath === '/login') {
    return <LoginPage />;
  }

  return (
    <div className={`min-h-screen ${currentUser.role === 'TECHNICIAN' ? 'bg-tech-bg text-slate-100' : 'bg-slate-50 text-slate-900'} flex flex-col font-sans transition-colors duration-200`}>
      {/* 3-ZONE ENTERPRISE TOP NAVBAR */}
      <Navbar />

      {/* MAIN WORKSPACE CONTENT */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {currentUser.role === 'ADMIN' ? (
          <AdminDashboard />
        ) : (
          <TechnicianDashboard />
        )}
      </main>

      {/* FOOTER */}
      <footer className={`border-t py-4 text-center text-xs ${
        currentUser.role === 'TECHNICIAN' 
          ? 'bg-tech-surface border-tech-border text-tech-muted' 
          : 'bg-white border-slate-200 text-slate-500'
      }`}>
        <p>
          NeuroMed AI — Dual-Role RAG Control Plane &amp; Knowledge Governance Platform © 2026
        </p>
      </footer>

      {/* STRICT ROLE BOUNDARY ACCESS DENIED MODAL */}
      <AccessDeniedModal />
    </div>
  );
}

