import React from 'react'
import { Play } from 'lucide-react'

const Navbar = () => {
    
  return (
    <>
    <nav className="fixed top-0 w-full z-50 bg-white/10 backdrop-blur-lg border-b border-white/20">
        <div className="max-w-7xl px-6 py-5">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-400 to-purple-500 rounded-lg flex items-center justify-center">
                <Play className="w-4 h-4 text-white" />
              </div>
              <span className="text-xl font-bold text-white">VideoSense</span>
            </div>
          </div>
        </div>
      </nav>
    </>
  )
}

export default Navbar