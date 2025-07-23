import React from 'react'
import { useState } from 'react';
import { ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom'

const Hero = () => {
    const navigate = useNavigate()
    const [showAuthModal, setShowAuthModal] = useState(false);
    const [authMode, setAuthMode] = useState('login');
    const [errorMsg, setErrorMsg] = useState('');

    const openAuth = (mode) => {
        setAuthMode(mode);
        setShowAuthModal(true);
    };

    const closeAuth = () => {
        setShowAuthModal(false);
    };

    const handleAuthSubmit = async (e) => {
        e.preventDefault();
        
        // Get form data
        const formData = new FormData(e.target);
        const data = Object.fromEntries(formData);
        
        try {
            const url = authMode === 'login' 
                ? 'http://videosense-production.up.railway.app/api/login' 
                : 'http://videosense-production.up.railway.app/api/register';
                
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                // Store user name in localStorage
                localStorage.setItem('userName', result.user.name);
                closeAuth();
                navigate('/home');
            } else {
                setErrorMsg(result.message); // Set error message
            }
        } catch (error) {
            console.log('Something went wrong!');
        }
    };

    return (
        <>
            <div className="pt-32 pb-20 px-6">
                <div className="max-w-7xl mx-auto text-center">
                    <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
                        Analyze Product Reviews
                        <span className="block bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                            With AI Power
                        </span>
                    </h1>
                    <p className="text-xl text-white/80 mb-12 max-w-3xl mx-auto leading-relaxed">
                        Discover what customers really think about your products. Our AI analyzes YouTube reviews 
                        and provides instant sentiment insights, saving you hours of manual video watching.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                        <button 
                            onClick={() => openAuth('register')}
                            className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-full text-lg font-semibold hover:shadow-2xl hover:scale-105 transition-all duration-300 flex items-center justify-center space-x-2"
                        >
                            <span>Start Analyzing</span>
                            <ChevronRight className="w-5 h-5" />
                        </button>
                        <button 
                            onClick={() => openAuth('login')}
                            className="px-8 py-4 bg-white/10 backdrop-blur-sm text-white rounded-full text-lg font-semibold border border-white/30 hover:bg-white/20 transition-all duration-300"
                        >
                            Sign In
                        </button>
                    </div>
                </div>
            </div>
            
            {showAuthModal && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl p-8 w-full max-w-md relative">
                        <button 
                            onClick={closeAuth}
                            className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
                        >
                            ✕
                        </button>
                        
                        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
                            {authMode === 'login' ? 'Welcome Back' : 'Create Account'}
                        </h2>

                        <form onSubmit={handleAuthSubmit} className="space-y-4">
                            {authMode === 'register' && (
                                <input 
                                    type="text" 
                                    name="name"
                                    placeholder="Full Name"
                                    required
                                    className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                                />
                            )}
                            
                            <input 
                                type="email" 
                                name="email"
                                placeholder="Email"
                                required
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                            />

                            <input 
                                type="password" 
                                name="password"
                                placeholder="Password"
                                required
                                className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                            />

                            {authMode === 'register' && (
                                <input 
                                    type="text" 
                                    name="company"
                                    placeholder="Company (optional)"
                                    className="w-full px-4 py-3 border border-gray-300 rounded-lg"
                                />
                            )}

                            <button 
                                type="submit"
                                className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg font-semibold"
                            >
                                {authMode === 'login' ? 'Sign In' : 'Create Account'}
                            </button>
                        </form>

                        <div className="mt-6 text-center">
                            <button 
                                onClick={() => {setAuthMode(authMode === 'login' ? 'register' : 'login')
                                setErrorMsg(''); 
                                }}
                                className="text-blue-600 hover:text-blue-800 font-semibold"
                            >
                                {authMode === 'login' ? 'Create Account' : 'Sign In Instead'}
                            </button>
                        </div>
                        {errorMsg && (
    <div className="text-red-600 text-center mt-2">{errorMsg}</div>
)}   
                    </div>
                
                </div>
            )}
        </>
    )
}

export default Hero